# A3 replay comparison: X500 recordings of 2026-10-23 (IN PROGRESS)

Roadmap A3: upstream hku-mars FAST_LIO (`a4743b09`, MappingUAV's drone config) versus this fork's
`lio_node` (`37941c0`, `lio_x500_matched.yaml`), replayed with `tools/eval/eval.py` on a Jetson
Orin Nano (the drone's computer), so the CPU and memory figures are the target platform's.

**Status: 3 of 10 runs done** (`still5` both stacks, `height` fork). The queue was stopped for a
shutdown; `run_all.sh` resumes it and skips finished runs. `summary.txt` is the table so far.

## Recordings (roadmap A2)
MID360 on the X500, props off, carried at hover height, raw `/livox/lidar` + `/livox/imu` only.
All five are complete: message counts match 10 Hz / 200 Hz to within one message.

| Bag | Duration | Content |
|---|---|---|
| `still5` | 304.7 s | untouched on a marked pad |
| `loop5` | 282.0 s | room loop, back to the pad, same orientation |
| `height` | 151.0 s | floor → knee → chest → overhead and back, twice; ends on the pad |
| `yaw` | 427.7 s | 90°, 180°, 360° turns up to ~2–3 s per turn, held overhead |
| `hallway` | 223.2 s | hallway out and back; starts and ends at a hallway spot, not the pad |

The bags (5.5 GB) are not in this repo: size, and whether the indoor scans may be public is
undecided.

## Matching
`lio_x500_matched.yaml` mirrors the upstream drone config (0.5 m downsampling, 3 iterations,
`det_range 40`, the same IMU noise and extrinsic). Not matchable by config: IMU init samples (fork
20, upstream 10 hard-coded) and the fork's always-on accumulated map (bounded at 2M points; it
shows in the fork's memory).

## Reading the numbers
`end_start_gap_m` compares the last pose with the **first** pose, which is taken before the filter
settles. On `still5` both stacks show a few cm of "gap" for a sensor that never moved, so part of
every gap is initial settling, not drift. Compare stacks on the same bag; treat absolute gaps with
that in mind.
