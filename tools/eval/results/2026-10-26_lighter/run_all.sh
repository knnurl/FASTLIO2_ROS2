#!/usr/bin/env bash
# Same-day evaluation of the lighter-lio commits on all five 2026-10-23 recordings.
source /opt/ros/humble/setup.bash
source /home/neric/colcon_ws/install/setup.bash
source /home/neric/fastlio2_ws/install/setup.bash
set -u
E=/home/neric/fastlio2_ws/src/FASTLIO2_ROS2/tools/eval
R=/home/neric/fastlio2_ws/src/FASTLIO2_ROS2
O=/home/neric/eval_runs/lighter
mkdir -p $O
build(){ (cd /home/neric/fastlio2_ws && MAKEFLAGS=-j4 colcon build --packages-select fastlio2 --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF > $O/build_$1.log 2>&1) || echo "$(date +%T) BUILD FAILED $1"; }
runbags(){ # $1 variant label, $2 lio stack, $3 config
  for b in still5 loop5 height yaw hallway; do
    o=$O/${1}_$b; [ -f $o/metrics.json ] && continue; rm -rf $o
    echo "$(date +%T) start $1/$b"
    (cd $E && python3 eval.py run --lio $2 --lio-config $3 --bag /home/neric/lidar_rec/2026-10-23_x500_$b --out $o > $o.console.txt 2>&1) || echo "$(date +%T) FAILED $1/$b"
  done; }
LEAN=/home/neric/fastlio2_ws/configs/lio_x500_lean.yaml
for v in base:2a403fb r1:3d4703f l4:b96314c; do
  name=${v%%:*}; sha=${v##*:}
  (cd $R && git checkout -q $sha -- fastlio2) && echo "$(date +%T) building $name ($sha)"; build $name
  runbags $name fork $LEAN
done
(cd $R && git checkout -q lighter-lio -- fastlio2); build final
runbags upstream upstream /home/neric/colcon_ws/src/drone_bringup/config/fast_lio_mid360_x500.yaml
echo "$(date +%T) ALL DONE"
