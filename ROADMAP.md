# Roadmap

This fork serves [MappingUAV](https://github.com/knnurl/MappingUAV): an X500 drone (PX4 v1.15,
Livox MID360, Jetson Orin Nano) flying indoors without GPS. There, LiDAR-inertial odometry is
PX4's position source: `px4_odom_bridge` feeds it into EKF2 as external vision. The priorities
are, in order:

1. **Never lose track.** A wrong or jumping pose reaches PX4's estimator.
2. **Map accurately.** The exploration mission's product is the 3D map.

**Parked:** the people filter (and its world-frame tracker fix), RandLA-Net semantic labelling,
the Spot / VLP-16 configs, and the closed-loop health prototype on `experimental/health-prototype`.

Status as of 2026-10-26.

## Known facts

- MappingUAV currently pins **upstream** [hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO)
  (ROS 2 branch, `a4743b09`), not this fork. Upstream's point-to-plane Jacobian is correct; the
  error fixed here in `949c8b6` came from this port.
- Upstream FAST-LIO publishes no health or degeneracy signal, so MappingUAV's
  `lio_health_guard` can only watch `/Odometry` for rate, jumps and NaNs.
- This fork adds what upstream lacks: global relocalization in a saved map (`localizer`), loop
  closure (`pgo`), map refinement (`hba`), a localizer status topic.
- Drone-mounted recordings exist since 2026-10-23 (A2): MID360 on the X500, props off, carried at
  hover height, about 23 minutes in five runs, all clean. The bags are not in this repo; results and
  notes are in `tools/eval/results/2026-10-23_x500/`. The earlier March recordings (`~/lidar_rec`)
  are handheld and waved about (roll and pitch up to 80°) and do not resemble the drone.
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

Recorded 2026-10-23: `still5`, `loop5`, `height` (floor to overhead and back; no stairs), `yaw`,
`hallway`. Stairs are still to record.

**A3. Decision:** keep upstream FAST-LIO on the drone and use this fork only for B4 and C, or
switch `px4_odom_bridge` to this fork's `lio_node`. Decided by A2's numbers: gap, height drift,
CPU and memory on the Orin Nano.

Results in (2026-10-26, `tools/eval/results/2026-10-23_x500/`), decision pending. With the fork
matched to the drone's upstream config: both stacks tracked all five recordings with no losses; the
fork's return gap is 0.4 to 3.5 cm smaller on four of five (worse by 0.5 cm on `height`); the fork
costs about twice the CPU (46 to 63 % of a core against 26 to 29 %, and 124 % against 61 % in the
hallway) and 2.4 times the memory (~440 MB against ~180 MB, partly its always-on accumulated map).
Neither recording made either stack degrade, so B1 and B3 still have no failure case to test on.

## B. Never lose track

**B1. A real health signal.** Publish at scan rate: the share of points matched to planes, the
residual, and a degeneracy measure from the filter's information matrix (the weakest direction).
`lio_health_guard` uses it to request a hover before the odometry visibly breaks. The same signal is
what MappingUAV's EV sanity guard needs for real drift detection
(`docs/px4_odom_bridge_ev_guard_spec.md` §9 there).

**B2. Odometry into EKF2 never jumps.** Relocalization and loop-closure corrections move only the
map frame (`map → odom`), never `/Odometry`. This is a hard rule, checked by a test.

**B3. Degenerate geometry.** Measure corridors and open halls first (A2). The 2026-10-23 hallway did
not degrade either stack, so a longer, plainer corridor or a large open hall is still needed.

There is no second horizontal source behind LIO at the moment. MappingUAV disabled optical flow in
EKF2 on 2026-10-13 (`EKF2_OF_CTRL 0`): in an A/B test it caused 28 to 35 cm excursions and a reset
during motion, most likely because the HereFlow's mounting rotation (`SENS_FLOW_ROT`) was never
measured. The TF02 rangefinder only aids height, and only low and slow. So when LIO degrades, EKF2
dead-reckons on the IMU and PX4 falls back to Altitude mode. That makes B1 more urgent. Re-enable flow
only after measuring `SENS_FLOW_ROT` and repeating the A/B test.

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

A1 (done) → A2 (recorded 2026-10-23; stairs outstanding) → A3 (results in, decision pending) → B1 and B2 → B3 → C1 to C3 → B4.
