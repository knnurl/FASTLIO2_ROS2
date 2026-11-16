# LIO health signal (roadmap B1) and the no-jump rule (B2)

Status 2026-11-16: built on branch `b1-health`, not merged. Unit tested; replayed on three of the
2026-10-23 X500 recordings. No recording degrades yet, so the thresholds below are provisional.

## What is published

`/fastlio2/lio_health` (`interface/msg/LioHealth`), one message per odometry sample with the same stamp.

| Field | Meaning |
|---|---|
| `valid` | False when no point matched a plane. The odometry for that stamp is then IMU prediction only. Before this signal that case was a line on stderr and nothing else. |
| `points`, `matched`, `match_ratio` | Points offered to the filter, points matched in the last iteration, their ratio. |
| `residual_rms` | RMS point-to-plane distance of the matched points (m). |
| `iterations` | Filter iterations used. |
| `translation_degeneracy`, `rotation_degeneracy` | 3 × smallest eigenvalue / trace of the 3×3 block of the measurement information matrix: 1 when every direction is equally constrained, 0 when one is not constrained at all. Unitless, independent of point count and of `lidar_cov_inv`. |
| `weakest_translation_dir` | World-frame direction that is least constrained. |

Computed in `fastlio2/src/map_builder/lio_health.h` (Eigen only) from the information matrix the filter
already builds in `LidarProcessor::updateLossFunc`; the last iteration of a scan stands. Cost: one 3×3
eigen-decomposition per iteration. No message is sent for the scan that initialises the map.

## What the recordings show

Replay at real time through `lio_node` with `configs/lio_x500_lean.yaml`, Jetson Orin Nano, 2026-11-16.

| Recording | Scans | Valid | match ratio (min / median) | residual rms (median / max) | translation degeneracy (min / p1 / median) | Weakest direction (lowest 10 %) | Largest odometry step |
|---|---|---|---|---|---|---|---|
| hallway | 2228 | all | 0.41 / 0.56 | 4.2 / 6.4 cm | 0.30 / 0.34 / 0.56 | vertical | 12.2 cm (1.2 m/s), 0 jumps |
| yaw | 4274 | all | 0.49 / 0.61 | 4.2 / 5.6 cm | 0.18 / 0.24 / 0.37 | x (room axis) | 6.3 cm, 0 jumps |
| loop5 | 2817 | all | 0.43 / 0.56 | 4.6 / 6.6 cm | 0.18 / 0.23 / 0.39 | x (room axis) | 8.8 cm, 0 jumps |

Readings:
- None of the three ever came near degenerate. That agrees with the roadmap's note that no recording has
  made either stack degrade, and it means the signal has been seen only in its healthy range.
- In the hallway the weakest direction is **vertical**, not along the corridor: the corridor had enough
  structure along its length, while the Mid-360 (−7° to +52°) sees little floor. Height is the axis the
  downward rangefinder backs up, which is the right redundancy to keep (`EKF2_RNG_CTRL 1`).
- Rotation was weakest on loop5 (minimum 0.085).
- `lio_node` CPU during these runs: 41 to 48 % of a core on hallway, 22 to 26 % on yaw and loop5. No
  same-day run without the health code was made, so this is not a measured overhead.

## Provisional thresholds (to be replaced when a failing recording exists)

Healthy envelope seen: translation degeneracy ≥ 0.18, match ratio ≥ 0.41, residual rms ≤ 6.6 cm.

| Level | Condition, held for 0.5 s (5 scans) | Intended reaction in MappingUAV |
|---|---|---|
| DEGRADED | `valid` false, or translation degeneracy < 0.08, or match ratio < 0.25, or residual rms > 10 cm | stop accepting goals, cancel the active behaviour (hover) |
| LOST | `valid` false for 1.0 s, or no health message for 1.0 s | treat as LIO silence: the watchdog lands |

These sit at roughly half the healthiest minimum. They are guesses until B3 supplies a long plain
corridor, an open hall, stairs and a props-on recording.

## B2: odometry never jumps

Holds by construction, and now by test (`tools/eval/test_no_jump_rule.py`):
- `lio_node` subscribes to IMU and LiDAR only, offers one service (save map), has no client and no TF
  listener: nothing outside can move its state;
- `pgo` and the localizer publish corrections only as a transform whose parent is the map frame, and never
  publish odometry.
`metrics.odometry_steps` measures the largest scan-to-scan step on a replay (table above: no jumps).
Not covered: a jump produced by the filter itself when it re-acquires after `valid` was false. That is what
`valid` and MappingUAV's EV guard (step rule) are for.

## How MappingUAV should use it (after the swap; not built)

1. `lio_health_guard` subscribes to `lio_health` instead of deriving health from odometry rate and jumps,
   applies the table above, and keeps publishing `/lio_health/state` (0 OK, 1 DEGRADED) so the planner and
   explorer need no change. Its present rate and jump checks stay as a second opinion.
2. `px4_odom_bridge` EV guard (its spec §9.3, drift detection): `valid` false is a reason to stop forwarding
   that sample; a run of invalid samples followed by a step is the incident shape of 2026-09-14.
3. One owner: the guard decides DEGRADED; the watchdog owns LOST. The bridge only refuses samples.
4. Explorer: prefer viewpoints away from `weakest_translation_dir` only if a venue ever shows degeneracy;
   not before.
