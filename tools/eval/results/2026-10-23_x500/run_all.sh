#!/usr/bin/env bash
# A3 replay queue: every 2026-10-23 X500 bag through upstream FAST-LIO and the fork.
# Skips runs that already have metrics.json; re-runs incomplete ones. Log: ~/eval_runs/run_all.log
source /opt/ros/humble/setup.bash
source /home/neric/colcon_ws/install/setup.bash
source /home/neric/fastlio2_ws/install/setup.bash
cd /home/neric/fastlio2_ws/src/FASTLIO2_ROS2/tools/eval || exit 1
for b in still5 loop5 height yaw hallway; do
  for s in upstream fork; do
    o=/home/neric/eval_runs/${b}_${s}
    [ -f $o/metrics.json ] && { echo "$(date +%T) skip $b/$s (done)"; continue; }
    rm -rf $o
    if [ $s = fork ]; then cfg=/home/neric/fastlio2_ws/configs/lio_x500_matched.yaml
    else cfg=/home/neric/colcon_ws/src/drone_bringup/config/fast_lio_mid360_x500.yaml; fi
    echo "$(date +%T) start $b/$s"
    if python3 eval.py run --lio $s --lio-config $cfg --bag /home/neric/lidar_rec/2026-10-23_x500_$b --out $o > $o.console.txt 2>&1
    then echo "$(date +%T) done $b/$s"; else echo "$(date +%T) FAILED $b/$s"; fi
  done
done
echo "$(date +%T) ALL DONE"
python3 /home/neric/eval_runs/summarize.py > /home/neric/eval_runs/A3_summary.txt 2>&1
