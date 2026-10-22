# Roadmap

This fork serves [MappingUAV](https://github.com/knnurl/MappingUAV): an X500 drone (PX4 v1.15,
Livox MID360, Jetson Orin Nano) flying indoors without GPS. There, LiDAR-inertial odometry is
PX4's position source: `px4_odom_bridge` feeds it into EKF2 as external vision. The priorities
are, in order:

1. **Never lose track.** A wrong or jumping pose reaches PX4's estimator.
2. **Map accurately.** The exploration mission's product is the 3D map.

**Parked:** the people filter (and its world-frame tracker fix), RandLA-Net semantic labelling,
the Spot / VLP-16 configs, and the closed-loop health prototype on `experimental/health-prototype`.

Status as of 2026-10-23.

## Known facts

- MappingUAV currently pins **upstream** [hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO)
  (ROS 2 branch, `a4743b09`), not this fork. Upstream's point-to-plane Jacobian is correct; the
  error fixed here in `949c8b6` came from this port.
- Upstream FAST-LIO publishes no health or degeneracy signal, so MappingUAV's
  `lio_health_guard` can only watch `/Odometry` for rate, jumps and NaNs.
- This fork adds what upstream lacks: global relocalization in a saved map (`localizer`), loop
  closure (`pgo`), map refinement (`hba`), a localizer status topic.
- The only recordings so far (`~/lidar_rec`, March 2026) are handheld and waved about: roll and
  pitch up to 80°, under 0.9 m/s, about 1 m of height change. They do not resemble the drone.
- Relocalization searches x, y and yaw only; every candidate sits at the map's average height
  (`z_offset_m` is hard-wired to 0, `localizer/src/localizers/global_localizer.cpp:113`). On the March data, a
  relocalization after a loss landed about 3 m from the first fix while the status said `TRACKING`.

## A. Choose the odometry for the drone, by measurement

**A1. Replay tool for both stacks** (`tools/eval/`). `eval.py run --lio fork|upstream` replays a
bag through either LIO and reports the return-to-start gap (split into horizontal and vertical),
the largest height deviation, and CPU and memory per node. Done; see `tools/eval/README.md`.

**A2. Gate 2-style recordings.** MID360 mounted on the X500, props off, carried at drone-like
attitude and speed; MappingUAV forbids recording raw `/livox/lidar` in flight. Record
`/livox/lidar` and `/livox/imu`:

- a 5-minute closed loop that returns to a marked pad (the gap is the drift; Gate 2 asks < 5 cm);
- 5 minutes held still (height drift; Gate 2 asks for none);
- height changes, and stairs if available;
- a long corridor or an open hall (geometry where LiDAR degrades);
- fast yaw turns.

**A3. Decision:** keep upstream FAST-LIO on the drone and use this fork only for B4 and C, or
switch `px4_odom_bridge` to this fork's `lio_node`. Decided by A2's numbers: gap, height drift,
CPU and memory on the Orin Nano.

## B. Never lose track

**B1. A real health signal.** Publish at scan rate: the share of points matched to planes, the
residual, and a degeneracy measure from the filter's information matrix (the weakest direction).
`lio_health_guard` uses it to request a hover before the odometry visibly breaks.

**B2. Odometry into EKF2 never jumps.** Relocalization and loop-closure corrections move only the
map frame (`map → odom`), never `/Odometry`. This is a hard rule, checked by a test.

**B3. Degenerate geometry.** Measure corridors and open halls first (A2). EKF2 also fuses the
HereFlow optical flow and the TF02 rangefinder; check that it holds position when LIO degrades.

**B4. Relocalization in a saved map** (repeat flights, recovery after a loss):

- search height too, seeded from the rangefinder or EKF2's altitude;
- an acceptance check that would have rejected the March 3 m relocalization.

Single-session exploration needs B4 less than B1 to B3, so it comes after them.

## C. Accurate maps

**C1. Map quality without ground truth.** Add to the replay tool: the loop gap plus wall and floor
thickness (how crisp planes are in the map).

**C2. Loop closure** (`pgo`) corrects drift in the map frame, judged on the loop recordings.
Subject to B2.

**C3. Post-flight refinement** (`hba`) of the exported map, after landing, feeding MappingUAV's
`map_export`.

**C4. Map hand-off.** Check what feeds MappingUAV's OctoMap (0.2 m voxels) and that it uses the
corrected map where it should.

## Order

A1 (done) → A2 (a recording session, by the operator) → A3 → B1 and B2 → B3 → C1 to C3 → B4.
