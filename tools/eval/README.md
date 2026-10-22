# Replay evaluation

Replays a recorded bag through `lio_node` (and `localizer_node` if you give it a map) and
measures the result. Use it to check whether a change makes mapping or localization better
or worse before trying it on the robot.

```bash
source ~/fastlio2_ws/install/setup.bash
cd ~/fastlio2_ws/src/FASTLIO2_ROS2/tools/eval

# LIO only; optionally save the map for later localization runs
python3 eval.py run --bag ~/lidar_rec/1 --out ~/eval_runs/bag1 --save-map ~/eval_runs/bag1.pcd

# LIO + localizer (global mode, fresh global search) against a map
python3 eval.py run --bag ~/lidar_rec/3 --map ~/eval_runs/bag1.pcd --out ~/eval_runs/bag3_loc

# compare two runs of the same bag, e.g. before/after a change
python3 eval.py compare ~/eval_runs/before ~/eval_runs/after
```

Each run writes `trajectory.csv`, `status.csv`, `offset.csv`, the node logs and `metrics.json`
to `--out`. The run uses ROS domain 87 (`--domain`), localhost only, and ignores
`CYCLONEDDS_URI`, so it cannot interfere with other ROS systems on the machine. The
localizer's pose file goes to the output folder, never to the real warm-start file.

## Metrics

| Metric | Meaning |
|---|---|
| `end_start_gap_m` | Distance between the last and first pose. On a recording that returns to its marked start, this is the drift. |
| `path_m`, `max_from_start_m` | Distance travelled; furthest point from the start. |
| `time_to_first_fix_s` | Seconds from the first status message to the first `TRACKING`. |
| `losses`, `recovery_s`, `unrecovered` | Times tracking was lost, how long each took to recover, and whether the run ended lost. |
| `tracking_fraction`, `dead_reckoning_s` | Share of time in `TRACKING`; time spent on LIO dead reckoning. |
| `offset_jumps`, `max_offset_jump_m` | Changes of the map→local transform larger than 0.2 m between updates. The first is the initial fix. Further jumps mean a relocalization or large tracking corrections; on a good map they should be rare. |
| `compare`: `rms_mm`, `max_mm`, `max_rot_deg` | Pose difference between two runs on matched timestamps. |

LIO replay is deterministic: two runs of the same build give identical trajectories, so any
difference in `compare` comes from the change. Localizer runs vary slightly between repeats
(its global search runs in the background), by up to a few centimetres.

## Recordings worth making

- A closed loop that returns to a marked start, so `end_start_gap_m` measures drift.
- A "kidnap" run: carry or cover the sensor and move it, to test relocalization.
- People walking around the sensor, for the people filter.
- A map-update session: localize in a mapped area, then drive into new space.

`metrics.py` has no ROS dependencies; `test_metrics.py` covers it (`python3 -m pytest tools/eval`).
