# A3 replay comparison: X500 recordings of 2026-10-23

Roadmap A3: upstream hku-mars FAST_LIO (`a4743b09`, MappingUAV's drone config) versus this fork's
`lio_node` (`37941c0`, `lio_x500_matched.yaml`), replayed with `tools/eval/eval.py` on a Jetson
Orin Nano (the drone's computer), so the CPU and memory figures are the target platform's.

**Status: complete, 10 of 10 runs (2026-10-26).** `summary.txt` is the table; `runs/` holds each
run's `metrics.json` and `trajectory.csv`.

## Results

`settled` is the return gap measured from the pose 8 s in (inside the opening still period), not
from the first pose: on `still5`, a sensor that never moved shows a 7 to 11 cm first-pose gap, which
is the filter settling, not drift. Both stacks replayed identical data, so their difference is the
meaningful number; absolute gaps also include how precisely the drone was set back on its mark.

| Bag | settled gap upstream | settled gap fork | CPU mean (p95), upstream | CPU mean (p95), fork | memory up / fork |
|---|---|---|---|---|---|
| still5 | 3.6 cm | 2.8 cm | 27 % (32) | 49 % (65) | 178 / 433 MB |
| loop5 | 13.5 cm | 10.0 cm | 28 % (32) | 63 % (85) | 181 / 447 MB |
| height | 5.7 cm | 6.2 cm | 26 % (30) | 46 % (64) | 178 / 412 MB |
| yaw | 9.5 cm | 8.9 cm | 29 % (32) | 62 % (85) | 181 / 436 MB |
| hallway | 4.2 cm | 3.8 cm | 61 % (70) | 124 % (161) | 183 / 442 MB |

CPU is % of one core on the Orin Nano. Findings:

- **Tracking:** both stacks tracked every recording to the end, no losses; pose counts equal scan
  counts on both.
- **Accuracy:** the fork is better on four of five, by 0.4 to 3.5 cm, and 0.5 cm worse on `height`.
  The only difference larger than likely placement error is `loop5` (3.5 cm).
- **Cost:** the fork uses about twice the CPU and 2.4 times the memory. Part of the memory is its
  always-on accumulated map; where the extra CPU goes has not been profiled.
- **Degeneracy:** the hallway degraded neither stack, so it is not yet a failure case for B1/B3.

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
