"""Mark episode boundaries on /episode/event.

    ros2 run jbl_nodes episode [--timeout S] [--expect N] start <method> <pose_id> <round>
    ros2 run jbl_nodes episode [--timeout S] [--expect N] end <success|fail|abort> [reason ...]
    ros2 run jbl_nodes episode [--timeout S] [--expect N] intervention [note ...]

"intervention" reports an operator intervention in the running episode: dock_judge fails it
(and, with auto_end, publishes "end fail intervention"), the controllers stop, and
trial_logger counts it and keeps the note. "end fail intervention [note]" still works where
no judge runs (trial_logger counts at least one intervention for it).

Publishes one std_msgs/String with reliable, transient_local QoS (depth 10). A fresh process
matches the running nodes one by one, and a node that misses the initial discovery burst
matches only at the next periodic announcement (Fast DDS: every 3 s; measured with the
simulation running on Jazzy: the last of four nodes up to 3 s after the first). Publishing at
once would hand the event to each node as it matches, seconds apart, so that dock_judge could
judge an episode fake_robot has not started yet. So the command first waits until no new
subscriber has matched for 3.5 s (or, with ``--expect N``, until N have), then publishes once
to all of them, and keeps the publisher until they acknowledge it; a node matching even later
still gets the transient-local sample while the publisher lives. ``--timeout`` (default 2 s)
bounds the wait for the first match (for all N with --expect); the first match of a fresh
process occasionally takes 2-3 s too, so use ``--timeout 5``, and in scripts that know how
many nodes listen, ``--expect``.

Exit status: 0 delivered (to at least N subscribers with --expect), 1 no subscriber matched
(nothing was published: run it again), 2 bad arguments, 3 fewer than --expect subscribers
matched (those got it: do not simply re-run a start).
"""

import argparse
import os
import re
import sys
import time
from typing import List, NamedTuple, Optional, Sequence

import rclpy
from rclpy.duration import Duration
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

EVENT_TOPIC = "/episode/event"
OUTCOMES = ("success", "fail", "abort")
MATCH_TIMEOUT_S = 2.0
SETTLE_S = 3.5  # no new subscriber for this long after the first match: assume all have matched
ACK_TIMEOUT_S = 1.0
EXIT_DELIVERED, EXIT_NO_SUBSCRIBER, EXIT_USAGE, EXIT_PARTIAL = 0, 1, 2, 3
_METHOD_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

USAGE = (
    "episode [--timeout S] [--expect N] start <method> <pose_id> <round>\n"
    "       episode [--timeout S] [--expect N] end <%s> [reason ...]\n"
    "       episode [--timeout S] [--expect N] intervention [note ...]" % "|".join(OUTCOMES)
)


def event_qos() -> QoSProfile:
    """QoS every /episode/event publisher and subscriber uses (durability must match)."""
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


class EpisodeEvent(NamedTuple):
    kind: str  # "start", "end" or "intervention"
    method: str = ""
    pose_id: int = -1
    round: int = -1
    outcome: str = ""
    reason: str = ""  # end: the reason; intervention: the note


def _non_negative_int(text: str, what: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise ValueError("%s must be an integer, got %r" % (what, text)) from None
    if value < 0:
        raise ValueError("%s must be >= 0, got %d" % (what, value))
    return value


def parse_event_args(args: Sequence[str]) -> EpisodeEvent:
    """Command-line words (or the words of a message) -> EpisodeEvent; ValueError if malformed."""
    if not args:
        raise ValueError("missing event kind (start, end or intervention)")
    kind, rest = args[0], list(args[1:])
    if kind == "start":
        if len(rest) != 3:
            raise ValueError("start takes exactly <method> <pose_id> <round>")
        method = rest[0]
        if not _METHOD_RE.match(method):
            raise ValueError("method must be one word of letters, digits, '_', '-' or '.', got %r" % method)
        return EpisodeEvent(
            "start",
            method=method,
            pose_id=_non_negative_int(rest[1], "pose_id"),
            round=_non_negative_int(rest[2], "round"),
        )
    if kind == "end":
        if not rest:
            raise ValueError("end takes <%s> [reason ...]" % "|".join(OUTCOMES))
        if rest[0] not in OUTCOMES:
            raise ValueError("outcome must be one of %s, got %r" % (", ".join(OUTCOMES), rest[0]))
        return EpisodeEvent("end", outcome=rest[0], reason=" ".join(" ".join(rest[1:]).split()))
    if kind == "intervention":
        return EpisodeEvent("intervention", reason=" ".join(" ".join(rest).split()))
    raise ValueError("event kind must be start, end or intervention, got %r" % kind)


def parse_episode_event(text: str) -> EpisodeEvent:
    """/episode/event payload -> EpisodeEvent; ValueError if malformed."""
    return parse_event_args(text.split())


def format_episode_event(event: EpisodeEvent) -> str:
    if event.kind == "start":
        return "start %s %d %d" % (event.method, event.pose_id, event.round)
    if event.kind == "intervention":
        return ("intervention %s" % event.reason).rstrip()
    return ("end %s %s" % (event.outcome, event.reason)).rstrip()


def _wait_for_subscribers(pub, deadline: float, expect: int, settle_s: float) -> int:
    """Wait until ``expect`` subscribers matched (by ``deadline``) or, without an expectation,
    until none has been added for ``settle_s`` after the first match; returns the number
    matched (0: nobody matched by ``deadline``)."""
    count, changed = 0, time.monotonic()
    while True:
        now = time.monotonic()
        n = pub.get_subscription_count()
        if n != count:
            count, changed = n, now
        if expect > 0:
            if count >= expect or now >= deadline:
                return count
        elif count == 0:
            if now >= deadline:
                return 0
        elif now - changed >= settle_s:
            return count
        time.sleep(0.01)


def publish_event(
    text: str,
    match_timeout_s: float = MATCH_TIMEOUT_S,
    args: Optional[List[str]] = None,
    expect: int = 0,
    settle_s: float = SETTLE_S,
) -> int:
    """Publish ``text`` once; return the number of subscribers that matched (0: not published)."""
    rclpy.init(args=args)
    node = rclpy.create_node("episode_cli_%d" % os.getpid())
    try:
        pub = node.create_publisher(String, EVENT_TOPIC, event_qos())
        if _wait_for_subscribers(pub, time.monotonic() + match_timeout_s, expect, settle_s) == 0:
            return 0
        pub.publish(String(data=text))
        if hasattr(pub, "wait_for_all_acked"):  # Galactic and newer
            if not pub.wait_for_all_acked(Duration(seconds=ACK_TIMEOUT_S)):
                print(
                    "episode: not every subscriber acknowledged '%s' within %.0f s" % (text, ACK_TIMEOUT_S),
                    file=sys.stderr,
                )
        else:  # Foxy has no ack query: leave the reliable writer time to deliver
            time.sleep(0.5)
        return pub.get_subscription_count()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    parser = argparse.ArgumentParser(prog="episode", usage=USAGE, description="Publish one /episode/event.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=MATCH_TIMEOUT_S,
        metavar="S",
        help="seconds to wait for the first subscriber (for N with --expect) (default %(default)s)",
    )
    parser.add_argument(
        "--expect",
        type=int,
        default=0,
        metavar="N",
        help="number of subscribers that must receive the event, e.g. from a script that knows "
        "which nodes run (default: wait until no new one matches for %.0f s)" % SETTLE_S,
    )
    parser.add_argument("event", nargs=argparse.REMAINDER, help="start ... | end ... | intervention ...")
    opts = parser.parse_args(rclpy.utilities.remove_ros_args(argv)[1:])
    try:
        if opts.timeout <= 0:
            raise ValueError("--timeout must be positive")
        if opts.expect < 0:
            raise ValueError("--expect must be >= 0")
        if any(word.startswith("--") for word in opts.event):
            raise ValueError("options go before the event words")
        text = format_episode_event(parse_event_args(opts.event))
    except ValueError as exc:
        parser.error(str(exc))
    matched = publish_event(text, opts.timeout, argv, expect=opts.expect)
    if matched == 0:
        print(
            "episode: no subscriber on %s within %.1f s; '%s' was not delivered"
            % (EVENT_TOPIC, opts.timeout, text),
            file=sys.stderr,
        )
        return EXIT_NO_SUBSCRIBER
    if matched < opts.expect:
        print(
            "episode: only %d of %d expected subscribers on %s within %.1f s got '%s'"
            % (matched, opts.expect, EVENT_TOPIC, opts.timeout, text),
            file=sys.stderr,
        )
        return EXIT_PARTIAL
    print("published '%s' to %d subscriber(s)" % (text, matched))
    return EXIT_DELIVERED


if __name__ == "__main__":
    sys.exit(main())
