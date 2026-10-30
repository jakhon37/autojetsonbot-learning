# v4 run log

> **SIMULATION ONLY.** Every step below ran `fake_robot`, the kinematic simulator, or analysed
> what it recorded. Nothing was measured on autoJetsonBot.

This file has three parts:

- **Timing log.** The wall-clock UTC start and end of each step of [PREREG.md](PREREG.md)
  section 9, with its exit status. It is bookkeeping kept at the coordinator's request and
  changes nothing that was recorded or analysed. (Section 12 of the PREREG lists `RUN_LOG.md`
  only after an infrastructure failure; this file exists for the timing log as well.)
- **Infrastructure failures and deviations** (sections 8 and 9): every infrastructure failure,
  with what its discarded attempt had scored, and every deviation from the commands as written.
- **Review edits.** What was changed in the write-up after the run, and why.

## Setup

- 2026-10-29T12:26:09Z: b72ba83 (the pre-registration) was committed. GitHub's repository
  activity for `jakhon37/autojetsonbot-learning` records its push to `main` (89588a8 → b72ba83)
  at 2026-10-29T12:26:22Z (push event 48228263080, read with
  `gh api repos/jakhon37/autojetsonbot-learning/activity` at 17:23Z).
- 2026-10-29T12:28:30Z: the setup line printed `b72ba83e9a553df3e72d81ef5a3f0c11e8a376db` (clean
  tree). Docker was up (`mirror.gcr.io/library/ros@sha256:8eed0a4669bbf43ee331878a4c39bd8ebddea8e3312a58b4fae8edb14526b617`),
  11 GB free on `/`, load average 2.30 1.90 2.03. `SIM_CPUSET=0-2`; another agent's process ran
  under `taskset -c 3` at nice 15 (but see "Shared host" below).

## Timing log

| Step | Start (UTC) | End (UTC) | EXIT | Notes |
|---|---|---|---|---|
| 1. Demonstrations, set A (`sim_work/v4_demos.out`) | 2026-10-29T12:28:57Z | 2026-10-29T12:45:24Z | 0 | 40 rows; no slot skipped, aborted or repeated; no node died. Load average 2.20 at the start, 5.24 at the end (another agent's C++ build ran under `taskset -c 3` at nice 15; threads of its ONNX Runtime test ran on cores 1–2 at 12:31:52–12:31:54, see "Shared host"). |
| 2. Demonstrations, set B (`sim_work/v4_62_demos.out`) | 2026-10-29T12:45:52Z | 2026-10-29T13:03:54Z | 0 | 40 rows (39 successes; spot 28 round 2 a timeout at 30.06 s); no slot skipped, aborted or repeated; no node died. Load average 3.31 at the start, 1.89 at the end. |
| 3. How the demonstrations ended (host) | 2026-10-29T13:04:16Z | 2026-10-29T13:04:23Z | 0 | "v4: 40 successes of 40 scheduled demonstrations, 40 of 40 slots scored"; "v4_62: 39 successes of 40 …, 40 of 40 slots scored". Gate passed (both ≥ 30). |
| 4a. Dataset and model, set A (`sim_work/v4_train.out`) | 2026-10-29T13:04:50Z | 2026-10-29T13:49:34Z | 0 | 40 episodes, 6,395 frames; train 32 / 5,062, val 8 / 1,333. Epochs of 390, 368, 374, 412 and 973 s (the last under the other agent's load; load average 4.74 at the end); best epoch 4; ONNX Runtime vs PyTorch 2.98e-07. |
| 4b. Dataset and model, set B (`sim_work/v4_62_train.out`) | 2026-10-29T13:49:56Z | 2026-10-29T14:29:59Z | 0 | 40 episodes (39 successes), 6,976 frames; train 30 / 5,195, val 9 / 1,481. Epochs of 735 (under the other agent's load), 394, 375, 370 and 387 s; best epoch 5; ONNX Runtime vs PyTorch 2.98e-07. |
| 5. Model checks and training records (host) | 2026-10-29T14:30:15Z | 2026-10-29T14:30:16Z | 0 | sha256 `ed36c9b6…` (`bc_v4`) and `ea29531c…` (`bc_v4_62`) match `sha256sum`; opset 13; parity 2.98e-07 for both. |
| 6. Before any rollout (host) | 2026-10-29T14:30:49Z | 2026-10-29T14:32:14Z | 0 | "P1, set A: full stops in the last 2 s: 454 / 6395 = 7.1 %". |
| 7. Evaluation, session 1 (`sim_work/v4_eval.out`) | 2026-10-29T14:32:38Z | 2026-10-29T16:03:52Z | 0 | Segments `clean_ref` 14:35:03–14:43:08, `noise` 14:43:08–15:06:53, `wide` 15:06:53–15:35:45, `hfov62` 15:35:45–16:03:12. Both completion checks passed (24 files; every `report.md` has the schedule line). No slot skipped, aborted or repeated; no node died. |
| 8. Evaluation, session 2 (`sim_work/v4_62_eval.out`) | 2026-10-29T16:04:36Z | 2026-10-29T16:19:50Z | 0 | Segment `hfov62` 16:06:31–16:19:34. Both checks passed (schedule 30 each for `bc_v4`, `bc_v4_62`, `servo`; the schedule line in `report.md`). No slot skipped, aborted or repeated; no node died. |
| 9. Tables (host) | 2026-10-29T16:20:19Z | 2026-10-29T16:20:22Z | 0 | `table` and `stops` both exited 0; `P3_stops.txt` is complete (no "INCOMPLETE" line). |
| 10. Write-up | 2026-10-29T16:20:22Z | 2026-10-29T16:32:27Z | — | `SUMMARY.md` drafted (see the deviation below). README: the v4 results subsection added, and three statements that said v4 had not run were updated ("three runs" became "four runs", the v4 protocol heading lost "not run yet", and the Limitations bullet on stopping was rewritten). |

## Infrastructure failures and deviations

**Infrastructure failures: none.**

- Every step ran once and completed, and every completion check of section 9 passed.
- No attempt was discarded and none was rerun.
- No slot was skipped, aborted or repeated, and there was no `fail controller_down`.
- No node died ("process has died" appears in no launch or servo log).

**DEVIATION (step 10).** The experimenter's environment refused to create `SUMMARY.md` (a
guard against agents writing report files). Its full text was handed to the coordinator. After
four reviews, a fixing step edited it ("Review edits" below); that step's own attempt to create
the file was refused by the same guard, so the reviewed text went back to the coordinator to
save as `results/sim_eval_v4_cleanstop/SUMMARY.md`. Nothing recorded or analysed depends on
this.

**Bookkeeping around the commands** (not deviations: the commands ran as written):

- UTC timestamps and exit statuses were printed around each step.
- In step 6, `|| rc=$?` collected the exit statuses.
- In step 9, the table's printed copy was sent to `/dev/null`. `--out` writes the same text to
  `SUMMARY_table.md`.
- `PYTHON` was the scratchpad virtual environment (Python 3.12.11), in place of the setup
  block's placeholder path.

**Observations (not failures):**

- **Provenance gap.** In session 1's `noise` segment, the provenance query for
  `servo_baseline.hfov_rad` returned "Node not found". The other values were reported, including
  `servo_baseline.stop_hysteresis: 0.03`. The servo's own start-up line in
  `sim_work/v4/eval_v2/jazzy/noise/logs/seg_20261029T144308Z.servo.log` reads "hfov 1.3960 rad
  … stop_hysteresis 0.030 m".
- **Shared host.** From the other agents' own logs and file times on this machine (not in git):
  - Steps 1–4: another agent's C++ builds and benchmarks ran under `taskset -c 3` at nice 15.
    Its thread-affinity log shows threads of its ONNX Runtime test on cores 1 and 2, inside
    `SIM_CPUSET`, at 12:31:52–12:31:54, during step 1; that process ended at 12:32:44, and the
    agent added a shim keeping ONNX Runtime on core 3 at 12:34. In the `demos_r1` bag, the
    safety gate missed 4, 0, 2 and 0 deadlines in the 30 s windows from 12:31:11 to 12:33:11, no
    more than later in that round (0–4). Training is unpinned, so two epochs slowed (973 s in
    step 4a, 735 s in 4b).
  - Session 1, `wide` segment (15:06:53–15:35:45): a second workflow (SmolVLA real-weight
    checks), whose rules forbade heavy compute until session 2 had ended, downloaded about
    0.9 GB of checkpoints and dataset files into `/dev/shm` (15:11–15:15), decoded dataset video
    frames (15:19 and 15:34), compiled a small `LD_PRELOAD` library (15:25) and imported
    `transformers` (15:31). Whether that work stayed on core 3 is not recorded. From 15:35 it
    wrote scripts and polled; during session 2 it only polled, and its log lists only the
    session's container (`jbl_eval_jazzy_20777`).
  - Spot checks during steps 7 and 8 (14:41, 15:39, 15:49 and 16:14 UTC) found nothing running
    beside the simulation container. They missed the `wide`-segment work above, and their
    output was not saved.
- **Slow inference.** Policy inference ran at p50 60–69 ms and p99 116–123 ms in both sessions,
  2–3 times v3's. Fallback zeros appeared in 37 of 130 policy trials (each `report.md`, and the
  validity check in `SUMMARY_table.md`). Section 6 of the PREREG anticipates uncontrolled load,
  and this is not an infrastructure failure under section 8. The slowdown was already there in
  `clean_ref`, before the second workflow started, and in session 2 (`SUMMARY.md`, exploratory
  3). The pilot ran only the servo pinned, so no policy arm had run pinned before session 1.
  Session 2's load average rose from 1.27 to 7.94 with only its own container running; this was
  not investigated.
- **`git.dirty: true` in `training/*/policy_meta.json`.** `train_bc.py` sets it from
  `git status --porcelain`, which also lists untracked files. Step 3's result files (written
  13:04:20–13:04:23) were untracked when training started (13:04:50 and 13:49:56). The stage
  provenance, which leaves the run's own result files out, records 0 changed paths.

## Review edits

Four reviews (numbers, protocol, claims, completeness) checked the write-up against the result
files, the bags and the logs. They found no wrong number in the primary results, the
comparisons, the validity check or the prediction verdicts. Nothing was rerun, and no data or
analysis output changed; the edits below are to the text only.

- **`SUMMARY.md`.** The handed-over text had sha256
  `0724a216502abfe3a4de40657f57bc26b35b721f59290e86de156b95ba592d1a` (with a final newline); the
  reviewed text has sha256 `34e968c79e8f547f55d3300d4cfab9ad8caa1db63ca0973bd14ed826599d3899`.
  The edits:
  - introduction: the commit and push times, the pilot, the step-10 deviation;
  - "What ran": the other work on the host (above) in place of "core 3, nice 15" and "nothing
    else running"; why the model records say `git.dirty: true`;
  - "Predictions": the P4 and P5 count verdicts give the counts with flagged successes counted
    as failures; C1 counted that way (`noise` 13 / 0, p = 0.0002; exploratory); P3's stop part
    against the successes and failures; what the pilot had already shown;
  - section 1: the latch and the spot-17 restart labelled exploratory, with their sources; the
    5 and 14 mid-episode full stops among the other stop frames, and the 3 among set B's 113
    mid-episode validation frames; the offline interpretation marked as one;
  - section 2: the every-request "Stop commanded" sensitivity line per condition (PREREG
    section 5); pose 1 described by its smallest requests; session 2's worse gate timing and
    load next to C2 and C3; the gate stalls;
  - section 3 and "Exploratory": causal wording softened; 56 (not 58) of 75 successes reached
    0.004 m/s (58 only after rounding); 16 of 18 failures, not all, stayed above 0.005 m/s;
    exploratory 3 rests on the timing evidence, not on the spot checks; new items 7–9 (final
    heading, per-pose tests, margin to the 30 s limit);
  - "What this shows" and "What this does not show": (a) answered directly; "clearly worse" and
    "still below the servo" tied to the poses; the lens defined as the simulated HFOV; the
    timing handicap of the policy arms; the v2/v3 judge;
  - "Run record" and "Files": how this file was saved; the image digest; what is not archived.
- **`README.md`.** The step-10 edits stay. The review added "an x86 CPU" (not "laptop") in the
  results banner and, in the v4 subsection, the deviation, BC v2's training command, a note that
  "Stop commanded" equals success in `clean_ref` and `noise`, the P4/P5 caveat, the poses behind
  the `wide` and C3 deficits, and a short "What it does not show". The Limitations bullet keeps
  "reliably": v3's `bc_v2` once requested 0.0013 m/s on the band's edge
  (`results/sim_eval_v3_bcfix/hfov62`, pose 1 round 3).
- **This file.** The push evidence, the shared-host record, the `git.dirty` note, the full list
  of step-10 README edits, and this section.
- **Not done.** The pinned container's inference timing was not benchmarked, and the bags and
  models were not archived.
- **Coordinator, after the review.** Copied the four files of `sim_work/v4_exploratory/` unchanged
  into [exploratory/](exploratory/) and changed three passages of `SUMMARY.md` to point there
  (the spot-17 source, the Exploratory introduction and the file list). No number changed.
  `SUMMARY.md` sha256 after that edit:
  `49f18a0d4adaf59c6d64535fe8e087b9b645856e7ed22f64da1bad66b8e4db95`.
- **Timing follow-up, after the review (exploratory, post hoc).** Added
  [exploratory/TIMING.md](exploratory/TIMING.md), which investigates the slow inference, with its
  scripts and raw outputs in [exploratory/timing/](exploratory/timing/). The bags and logs were
  only read. The benchmarks ran on 29 October 2026, 21:17–22:04 UTC, with repeats after that
  follow-up's review at 23:27–23:47 UTC, after this run, in fresh containers. Appended item 10
  to the Exploratory section of `SUMMARY.md`, one line pointing to it; no existing text of
  `SUMMARY.md` changed, and no number or verdict changed. `SUMMARY.md` sha256 after that edit:
  `ff0f6ec9db2083ab7a70d02984ff87150041ff948c8e1bcd6d45e987e0663642`.
