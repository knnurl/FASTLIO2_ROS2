#!/bin/bash
# Builds the whole workspace and runs the unit tests on ROS 2 Humble.
# Runs inside ros:humble-ros-base-jammy with this repo at $WS/src/FASTLIO2_ROS2,
# both in GitHub Actions and locally:
#   docker run --rm -v "$PWD":/ws/src/FASTLIO2_ROS2 ros:humble-ros-base-jammy \
#     bash /ws/src/FASTLIO2_ROS2/.github/ci/build_and_test.sh
set -euo pipefail

WS=${WS:-/ws}
JOBS=${JOBS:-4}
LIVOX_SDK2_REF=6a94015        # Livox-SDK2, 2024-02-18
LIVOX_DRIVER_REF=6b9356c      # livox_ros_driver2, 2024-09-11

export DEBIAN_FRONTEND=noninteractive
# Launchpad (GTSAM PPA) has occasional 503s; let apt retry.
echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries
apt-get update -qq
apt-get install -y -qq git cmake build-essential python3-rosdep \
    software-properties-common ros-humble-sophus > /dev/null
# GTSAM 4.1 for pgo/hba, from the borglab PPA (as on the development machine).
add-apt-repository -y ppa:borglab/gtsam-release-4.1 > /dev/null
apt-get update -qq
apt-get install -y -qq libgtsam-dev libgtsam-unstable-dev > /dev/null

# Livox SDK2 and its ROS 2 driver are not in rosdep; build them from pinned commits.
if [ ! -d /tmp/Livox-SDK2 ]; then
    git clone -q https://github.com/Livox-SDK/Livox-SDK2.git /tmp/Livox-SDK2
fi
git -C /tmp/Livox-SDK2 checkout -q "$LIVOX_SDK2_REF"
cmake -S /tmp/Livox-SDK2 -B /tmp/Livox-SDK2/build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build /tmp/Livox-SDK2/build -j"$JOBS" > /dev/null
cmake --install /tmp/Livox-SDK2/build > /dev/null
ldconfig

DRIVER=$WS/src/livox_ros_driver2
if [ ! -d "$DRIVER" ]; then
    git clone -q https://github.com/Livox-SDK/livox_ros_driver2.git "$DRIVER"
fi
git -C "$DRIVER" checkout -q "$LIVOX_DRIVER_REF"
cp "$DRIVER/package_ROS2.xml" "$DRIVER/package.xml"
rm -rf "$DRIVER/launch" && cp -r "$DRIVER/launch_ROS2" "$DRIVER/launch"

# lidar_semantic's torch/scipy runtime deps are not needed to build or test.
rosdep update -q > /dev/null || true
rosdep install --from-paths "$WS/src" --ignore-src -y -r -q --rosdistro humble \
    --skip-keys "GTSAM" > /dev/null || true

set +u
source /opt/ros/humble/setup.bash
set -u
cd "$WS"
colcon build --event-handlers console_direct- \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DROS_EDITION=ROS2 -DHUMBLE_ROS=humble \
    --parallel-workers "$JOBS"

# Unit tests only: the packages keep the upstream ament linters (uncrustify,
# cppcheck, ...) enabled, and the existing code does not pass them.
colcon test --packages-select fastlio2 localizer lidar_people_filter \
    --event-handlers console_direct- --ctest-args -LE linter
colcon test-result --verbose
