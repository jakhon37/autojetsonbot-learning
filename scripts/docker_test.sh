#!/usr/bin/env bash
# Build and test the ROS 2 packages in clean Foxy / Humble / Jazzy containers.
#
#   bash scripts/docker_test.sh [foxy humble jazzy]     (default: all three)
#   PKGS="jbl_safety" bash scripts/docker_test.sh foxy   (only these packages)
#
# Uses the official ros:<distro>-ros-base images (pulled from mirror.gcr.io when Docker Hub
# rate-limits) and installs only Ubuntu packages (python3-opencv, python3-pytest, numpy).
# Works behind an HTTPS proxy: set HTTPS_PROXY and, if TLS is re-terminated, CA_BUNDLE.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
DISTROS=${*:-foxy humble jazzy}
PKGS=${PKGS:-}
REGISTRY=${REGISTRY:-mirror.gcr.io/library}
CA_BUNDLE=${CA_BUNDLE:-/root/.ccr/ca-bundle.crt}
mkdir -p build_logs

run_one() {
  local d=$1
  local proxy_args=()
  if [ -n "${HTTPS_PROXY:-}" ]; then
    proxy_args+=(--network host -e "HTTPS_PROXY=$HTTPS_PROXY" -e "https_proxy=$HTTPS_PROXY")
  fi
  if [ -f "$CA_BUNDLE" ]; then
    proxy_args+=(-v "$CA_BUNDLE:/usr/local/share/ca-certificates/extra-ca.crt:ro")
  fi
  mkdir -p "build_logs"
  docker run --rm "${proxy_args[@]}" -e "PKGS=$PKGS" -v "$REPO:/repo:ro" "$REGISTRY/ros:$d-ros-base" bash -c '
    set -eo pipefail  # else a failing "colcon test | tail" returns the status of tail and passes
    update-ca-certificates >/dev/null 2>&1 || true
    # apt over HTTPS only (plain-HTTP mirrors may be blocked by the proxy); ROS apt repo not needed
    sed -i "s|http://\(archive\|security\|ports\).ubuntu.com|https://\1.ubuntu.com|g" /etc/apt/sources.list 2>/dev/null || true
    for f in /etc/apt/sources.list.d/*; do [ -f "$f" ] && sed -i "s|http://\(archive\|security\|ports\).ubuntu.com|https://\1.ubuntu.com|g" "$f"; done
    rm -f /etc/apt/sources.list.d/ros2*
    apt-get update -qq >/dev/null 2>&1
    apt-get install -y -qq python3-opencv python3-pytest python3-numpy >/dev/null 2>&1
    . /opt/ros/$ROS_DISTRO/setup.sh
    mkdir -p /ws/src && cp -r /repo/ros2_ws/src/* /ws/src/ && cd /ws
    SEL=""; [ -n "$PKGS" ] && SEL="--packages-select $PKGS"
    colcon build $SEL --event-handlers console_cohesion+ --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -30
    test "${PIPESTATUS[0]}" -eq 0
    . install/setup.sh
    colcon test $SEL --event-handlers console_cohesion+ --return-code-on-test-failure 2>&1 | tail -60 || { colcon test-result --verbose; exit 1; }
    colcon test-result --verbose | tail -15
  ' 2>&1 | tee "build_logs/$d.log"
  return "${PIPESTATUS[0]}"
}

status=0
for d in $DISTROS; do
  echo "================= $d ================="
  if run_one "$d"; then echo "$d: PASS"; else echo "$d: FAIL"; status=1; fi
done
exit $status
