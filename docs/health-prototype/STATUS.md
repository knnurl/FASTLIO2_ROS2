# Health-layer prototype — status

This branch is a snapshot of `~/fastlio3_ws/src` as it stood on 2026-04-07. It saves the work so it is not lost. It is **not** maintained code: the maintained stack is on `main`.

## What it is

An agent implemented [`IMPLEMENTATION_PROMPT.md`](IMPLEMENTATION_PROMPT.md), an 8-task plan for a "closed feedback loop" around the mapping stack. The result:

- `fastlio2_health/` (about 2,200 lines): `map_health_monitor` (quality and staleness grids, the HBA → merge → verify → rollback state machine), `voxel_merge_node` (three merge policies), `PoseGraphPartitioner`, `MapStalenessTracker`, and six message types.
- `fastlio2_validation/` (about 700 lines): `map_validator`, which computes ATE (Horn alignment) and RPE, with baseline and compare services.
- Hooks in `fastlio2/`, `pgo/`, `hba/` and the GUI, plus `fastlio2_bringup/launch/full_system.launch.py`.

The branch starts from `7ad5691`, the GitHub state the agent cloned. `git diff 7ad5691 experimental/health-prototype` shows exactly what the prototype changed.

## State

- **Builds:** every package compiled on 2026-04-07.
- **Tests:** four test files exist (`fastlio2_health/test/`, `fastlio2_validation/test/`), but `colcon test` was never run.
- **Runs:** there is no record of the full system being run end to end.

## Known problems

1. **Built on an early localizer.** The snapshot's `localizer/src/localizer_node.cpp` (517 lines) can only re-run ICP from the identity pose (line 336). The localizer on `main` (1,568 lines, July 2026) adds the IDLE/SEARCHING/TRACKING/LOST state machine, a background NDT+ICP grid search and pose persistence. None of the health hooks use the localizer.
2. **Wrong signals.** `/fsm/state` is the LIO builder status (`IMU_INIT`/`MAP_INIT`/`MAPPING`, `fastlio2/src/lio_node.cpp:307`), not a localizer state. `/fsm/icp_fitness` is the LIO inlier ratio (`lidar_processor.cpp:230`, higher is better), not an ICP fitness.
3. **Inverted rollback.** `map_health_monitor.cpp:570` rolls back when `post_mean > pre_mean * (1 - improvement_threshold)`. That formula assumes lower is better, but this fitness is higher-is-better. So merges that keep or improve quality are rolled back, and merges that make it more than 5% worse are committed.
4. **Validation result ignored.** The ATE comparison request at `map_health_monitor.cpp:575` is sent, but its result is never read.
5. **Whole-map replacement mid-run.** HBA output is stamped `frame_id = "map"` (`hba/src/hba_node.cpp:285`), and `lio_node` swaps its entire ikd-tree for the merged cloud (`lio_node.cpp:327`) with no frame check. That risks a jump in odometry.
6. **Premises in the prompt were wrong.** It assumed components that `main` does not have in that form: a 4-state localiser named UNINITIALISED/LOCALISING/LOCALISED/DEGRADED, an eigenvalue-based degenerate-scan guard, and an HBA that patches the live map. Much of the glue targets those assumed interfaces.

## Reuse

`fastlio2_validation` is the part most worth carrying over: it is self-contained and runs without ground truth (RPE only). The health monitor should come back only after it has been rewired to the current localizer and run report-only first.
