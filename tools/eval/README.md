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

# upstream hku-mars FAST_LIO (as pinned by MappingUAV) on the same bag, with its config
python3 eval.py run --lio upstream --lio-config ~/MappingUAV/src/drone_bringup/config/fast_lio_mid360_x500.yaml \
    --bag ~/lidar_rec/2 --out ~/eval_runs/bag2_upstream
```

`--lio upstream` needs the `fast_lio` package built and sourced (it records `/Odometry`); the
default `--lio fork` runs this repo's `lio_node` (`/fastlio2/lio_odom`). `--map` and `--save-map`
work with the fork only. When comparing the two stacks, match their downsampling and
iteration settings first: MappingUAV's config filters at 0.5 m, this repo's `lio.yaml` at
0.1 m scans and a 0.05 m map, which changes both accuracy and CPU.

Each run writes `trajectory.csv`, `status.csv`, `offset.csv`, the node logs and `metrics.json`
to `--out`. The run uses ROS domain 87 (`--domain`), localhost only, and ignores
`CYCLONEDDS_URI`, so it cannot interfere with other ROS systems on the machine. The
localizer's pose file goes to the output folder, never to the real warm-start file.

## Metrics

| Metric | Meaning |
|---|---|
| `end_start_gap_m` | Distance between the last and first pose. On a recording that returns to its marked start, this is the drift. |
| `path_m`, `max_from_start_m` | Distance travelled; furthest point from the start. |
| `xy_gap_m`, `z_gap_m`, `z_max_dev_m` | The return-to-start gap split into horizontal and vertical, and the largest height deviation from the start. MappingUAV Gate 2: return-to-start < 5 cm, no height drift over 5 minutes. |
| `lio.resources` | CPU (% of one core, as `top` shows) and peak memory of the LIO node, sampled during playback. Measured on the machine that runs the replay, not on the Orin Nano. |
| `time_to_first_fix_s` | Seconds from the first status message to the first `TRACKING`. |
| `losses`, `recovery_s`, `unrecovered` | Times tracking was lost, how long each took to recover, and whether the run ended lost. |
| `tracking_fraction`, `dead_reckoning_s` | Share of time in `TRACKING`; time spent on LIO dead reckoning. |
| `offset_jumps`, `max_offset_jump_m` | Changes of the map→local transform larger than 0.2 m between updates. The first is the initial fix. Further jumps mean a relocalization or large tracking corrections; on a good map they should be rare. |
| `compare`: `rms_mm`, `max_mm`, `max_rot_deg` | Pose difference between two runs on matched timestamps. |

LIO replay is deterministic: two runs of the same build give identical trajectories, so any
difference in `compare` comes from the change. Localizer runs vary slightly between repeats
(its global search runs in the background), by up to a few centimetres.

## Recordings worth making

For the drone (see [`ROADMAP.md`](../../ROADMAP.md), A2): MID360 mounted on the X500, props off,
carried at drone-like attitude and speed. MappingUAV does not record raw `/livox/lidar` in
flight. Record `/livox/lidar` and `/livox/imu`:

- a 5-minute closed loop back to a marked pad (`end_start_gap_m` is the drift);
- 5 minutes held still (`z_max_dev_m` is the height drift);
- height changes, and stairs if available;
- a long corridor or an open hall;
- fast yaw turns.

`metrics.py` has no ROS dependencies; `test_metrics.py` covers it (`python3 -m pytest tools/eval`).
