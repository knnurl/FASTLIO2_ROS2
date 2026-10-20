# Agent Implementation Prompt — FAST-LIO2 ROS2 Closed-Loop Mapping System

## Target Repository
`https://github.com/knnurl/FASTLIO2_ROS2/tree/main`

Clone this repo and work directly on it. All changes must be ROS 2 Humble-compatible, C++17, and build cleanly with `colcon build --symlink-install`.

  colcon build --symlink-install --allow-overriding interface                   

---

## Context: What Already Exists

This repo is a FAST-LIO2 ROS 2 wrapper built around a **Livox MID-360** primary sensor and **Velodyne VLP-16** secondary sensor. The following components are already implemented as separate nodes/packages — treat them as stable, production-grade building blocks. **Do not refactor their internals unless explicitly required by an integration task below.**

| Component | Description |
|---|---|
| `fast_lio` core | FAST-LIO2 LiDAR-inertial odometry, ikd-Tree map, publishes `/cloud_registered`, `/Odometry`, `/path` |
| `laserMapping` node | Main FAST-LIO2 node in `src/laserMapping.cpp` — processes pointcloud + IMU, outputs pose and dense map |
| **FSM localiser** | 4-state Finite State Machine (`UNINITIALISED → LOCALISING → LOCALISED → DEGRADED`) managing ICP confidence and recovery; publishes ICP fitness score and FSM state |
| **Sensor-aware people filter** (`lidar_people_filter`) | Two-stage pipeline: DBSCAN v2 with range-adaptive clustering + temporal voxel consistency tracking; runs strictly downstream of FAST-LIO2 outputs |
| **Loop closure detector** | Background thread scanning pose graph for geometric candidates; produces match score and candidate pair (node_i, node_j) |
| **Subgraph HBA** (Hierarchical Bundle Adjustment) | Pose graph optimiser operating on a subgraph; takes a set of node IDs, outputs corrected poses; already integrated with ikd-Tree |
| **Hot-reload merge** | Applies corrected PCD sections into the live map without restarting the node; receives a corrected pointcloud + transform and patches the ikd-Tree |
| **Degenerate scan guard** | Analyses scan eigenvalue distribution; flags scans as `DEGENERATE` when the environment is geometrically uninformative (corridors, open spaces); currently suppresses ICP localisation on flagged scans |
| **Pose persistence** | Saves/loads pose graph to disk; includes staleness timestamps per pose node |
| **PyQt5 GUI** (`fastlio2_gui` or similar) | Operator dashboard showing FSM state, ICP fitness, map sections, HBA status |
| **`fastlio2_bringup`** | Launch package with YAML configs for MID-360 and VLP-16 profiles |

---

## What Is Missing: The Closed Feedback Loop

The components above are **open-loop** — they do not feed quality information back to influence the system's own behaviour. Your task is to close this loop by implementing the following features in priority order.

---

## Implementation Tasks

### TASK 1 — `map_health_monitor` Node (Core Coordinator) ★ HIGHEST PRIORITY

Create a new ROS 2 node `map_health_monitor` in a new package `fastlio2_health`. This is the architectural centrepiece that transforms the repo from a component collection into a self-healing system.

**Responsibilities:**
- Subscribe to ICP fitness scores, loop closure events, HBA completion signals, and merge completion signals
- Maintain a **spatial quality grid** — a 2D grid (configurable cell size, default 10m × 10m) where each cell stores:
  - `mean_icp_fitness` (rolling window, last N=50 scans localised in that cell)
  - `last_loop_closure_correction_magnitude` (metres, from most recent HBA affecting this cell)
  - `last_hba_timestamp`
  - `staleness_score` (float 0.0–1.0, computed from time since last confident localisation)
- Decide when any cell's degradation crosses `degradation_threshold` (configurable, default `0.4` on normalised scale)
- Orchestrate the **HBA → merge → verify** cycle (see Task 2) when threshold is crossed
- Expose spatial quality grid as a `nav_msgs/OccupancyGrid` on `/map_health/quality_grid` for RViz visualisation
- Publish health summary on `/map_health/status` as a custom message `MapHealthStatus.msg`
- Expose its internal state to the PyQt5 GUI via a `/map_health/gui_state` topic

**`MapHealthStatus.msg`:**
```
std_msgs/Header header
float32 global_health_score          # 0.0 (degraded) to 1.0 (healthy)
int32 num_degraded_cells
int32 num_cells_pending_hba
int32 num_cells_pending_merge
string current_phase                 # MONITORING | HBA_RUNNING | MERGING | VERIFYING | ROLLBACK
geometry_msgs/Point worst_cell_centre
float32 worst_cell_score
```

**Subscriptions:**
```
/fsm/icp_fitness          → std_msgs/Float32
/fsm/state                → std_msgs/String  (or your existing FSM state msg type)
/loop_closure/event       → (see Task 3 for message definition)
/hba/completion           → fastlio2_health/HBACompletion
/map_merge/completion     → fastlio2_health/MergeCompletion
/Odometry                 → nav_msgs/Odometry  (to know which cell the robot is currently in)
```

**Publications:**
```
/map_health/quality_grid  → nav_msgs/OccupancyGrid
/map_health/status        → fastlio2_health/MapHealthStatus
/map_health/gui_state     → std_msgs/String (JSON blob for GUI)
/hba/trigger              → fastlio2_health/HBATrigger
/map_merge/trigger        → fastlio2_health/MergeTrigger
```

---

### TASK 2 — Automated HBA → Merge → Verify Pipeline ★ HIGHEST PRIORITY

Wire the existing loop closure detector, HBA, and hot-reload merge into an automated pipeline driven by `map_health_monitor`. Currently these three components require manual intervention. This task automates them.

**Pipeline phases (state machine within `map_health_monitor`):**

```
MONITORING
    → (degraded cell detected OR strong loop closure received)
TRIGGER_HBA
    → publish /hba/trigger with affected node IDs (from spatial partition, see Task 5)
    → wait for /hba/completion
HBA_RUNNING
    → on /hba/completion received
MERGING
    → snapshot current ICP fitness baseline (mean over last K=20 scans, configurable)
    → publish /map_merge/trigger with corrected PCD + transforms
    → wait for /map_merge/completion
VERIFYING
    → collect ICP fitness over next V=30 scans (configurable)
    → compute post-merge mean fitness
    → if post_mean > pre_mean * (1 - improvement_threshold):  # improvement_threshold default 0.05
        → ROLLBACK: publish /map_merge/rollback, restore previous map snapshot
    → else:
        → COMMIT: update quality grid baseline for affected cells
        → return to MONITORING
ROLLBACK
    → log rollback event with full diagnostics
    → mark affected cells as NEEDS_REVIEW in quality grid
    → return to MONITORING
```

**Define these messages in `fastlio2_health/msg/`:**

`HBATrigger.msg`:
```
std_msgs/Header header
int32[] node_ids
geometry_msgs/Point cell_centre
float32 trigger_fitness_score
string trigger_reason   # LOOP_CLOSURE | DEGRADATION | SCHEDULED
```

`HBACompletion.msg`:
```
std_msgs/Header header
int32[] node_ids
bool success
float32 mean_correction_magnitude
sensor_msgs/PointCloud2 corrected_pcd
geometry_msgs/TransformStamped[] corrected_transforms
```

`MergeTrigger.msg`:
```
std_msgs/Header header
sensor_msgs/PointCloud2 incoming_pcd
geometry_msgs/TransformStamped transform
string merge_policy   # TIMESTAMP_PRIORITY | DENSITY_PRIORITY | CONFIDENCE_WEIGHTED
float32 icp_fitness_weight   # used only for CONFIDENCE_WEIGHTED
```

`MergeCompletion.msg`:
```
std_msgs/Header header
bool success
int32 points_added
int32 points_removed
float32 overlap_ratio
```

---

### TASK 3 — Loop Closure → HBA Signal + Degenerate Guard Suppression

**3a. Loop closure event message**

Define `LoopClosureEvent.msg` in `fastlio2_health/msg/`:
```
std_msgs/Header header
int32 node_i
int32 node_j
float32 match_score        # 0.0–1.0
float32 correction_magnitude_m
geometry_msgs/TransformStamped relative_transform
bool accepted              # whether the closure passed the acceptance threshold
```

Update the existing loop closure detector to publish `LoopClosureEvent` on `/loop_closure/event` on every detection attempt (both accepted and rejected closures, using `accepted` flag to distinguish).

**3b. Degenerate guard → loop closure suppression**

Extend the existing degenerate scan guard so that when a scan is flagged `DEGENERATE`:
- Set a `degenerate_suppression_flag` in a shared parameter or latched topic `/degenerate_guard/suppress_loop_closure` (`std_msgs/Bool`)
- The loop closure detector subscribes to this topic and **skips offering the current scan as a loop closure candidate** while the flag is true
- Log suppressed candidates at DEBUG level with the scan's eigenvalue distribution

This prevents geometrically uninformative scans (featureless corridors) from generating false loop closures that would trigger spurious HBA runs.

---

### TASK 4 — Voxel-Weighted Merge Strategy

Extend the hot-reload merge component to support three merge policies, selectable at runtime via the `MergeTrigger` message's `merge_policy` field:

**`TIMESTAMP_PRIORITY`**
When incoming and existing voxels overlap, the incoming (newer) scan's points survive. Discard existing points in the overlap region.

**`DENSITY_PRIORITY`**
For each overlapping voxel (use the ikd-Tree's existing voxel resolution), the side with more points per unit volume wins. Merge outcome is determined per-voxel, not globally.

**`CONFIDENCE_WEIGHTED`**
Weight point survival probability by `icp_fitness_weight` from the `MergeTrigger`. A voxel from the incoming scan survives with probability proportional to `(1 - incoming_fitness_score)` (lower fitness score = higher confidence = higher survival weight). Blend the point populations rather than hard-cut.

Add `merge_policy` as a parameter in `fastlio2_bringup` launch files with default `TIMESTAMP_PRIORITY`. Add a runtime-reconfigurable ROS 2 parameter so operators can switch policy without restarting the node.

---

### TASK 5 — Spatial Pose Graph Partitioning

For large environments, HBA on the full pose graph is O(n²) in node count. Implement spatial partitioning so HBA only operates on the subgraph containing the triggered loop closure.

**Implementation:**

Create a `PoseGraphPartitioner` class (can be a utility class inside `fastlio2_health` or a standalone library):

```cpp
class PoseGraphPartitioner {
public:
    // Cell size configurable, default 50.0m
    explicit PoseGraphPartitioner(double cell_size_m = 50.0);
    
    // Assign node to a spatial cell based on its position
    void insertNode(int node_id, const Eigen::Vector3d& position);
    
    // Get all node IDs in the cell containing position p,
    // plus nodes in adjacent cells (for boundary loop closures)
    std::vector<int> getSubgraphNodes(const Eigen::Vector3d& loop_closure_position,
                                       int radius_cells = 1) const;
    
    // Remove node (on pose graph pruning)
    void removeNode(int node_id);
};
```

Integrate this into the HBA trigger logic in `map_health_monitor`:
- When a loop closure fires, use `getSubgraphNodes()` centred on the loop closure position to get the relevant node IDs
- Pass only those IDs in `HBATrigger.node_ids`
- Log the subgraph size vs total graph size at INFO level so the operator can verify partitioning is working

Add `partition_cell_size_m` and `partition_radius_cells` as parameters in `fastlio2_bringup`.

---

### TASK 6 — Staleness-Aware Map Sections

The existing pose persistence code already tracks staleness timestamps per pose node. Extend this concept to map regions.

**Implementation:**

Add a `MapStalenessTracker` within `map_health_monitor` that:

- Maintains a map from `(cell_x, cell_y)` → `MapRegionMetadata`:
  ```cpp
  struct MapRegionMetadata {
      rclcpp::Time last_confident_localisation;  // last time ICP fitness was above threshold in this cell
      std::string session_id;                     // which mapping session populated this cell
      float staleness_score;                      // 0.0 fresh, 1.0 stale
      bool flagged_for_hba;
  };
  ```
- Updates `last_confident_localisation` every time an `/fsm/icp_fitness` message arrives above `confidence_threshold` (default 0.7) and the robot is in that cell
- Computes `staleness_score` as `min(1.0, elapsed_seconds / staleness_decay_s)` where `staleness_decay_s` is configurable (default 300s)
- When `staleness_score > staleness_hba_threshold` (default 0.8) in any cell, elevates that cell in the HBA queue — it gets processed next time HBA is triggered, ahead of less-stale cells

Expose a `/map_health/staleness_grid` as `nav_msgs/OccupancyGrid` (separate from the quality grid) for RViz visualisation.

---

### TASK 7 — ATE/RPE Validation Node

Create a new package `fastlio2_validation` with a node `map_validator` that computes standard trajectory accuracy metrics.

**Node inputs:**
```
/path                    → nav_msgs/Path  (FAST-LIO2 estimated trajectory)
/ground_truth/path       → nav_msgs/Path  (optional, from external reference or loaded from file)
/ground_truth/pcd_file   → ROS 2 parameter (string path to .pcd reference map, optional)
```

**Metrics computed:**

*Absolute Trajectory Error (ATE):*
- Align estimated trajectory to ground truth using Horn's method (closed-form SE3)
- Compute per-pose translation error after alignment
- Publish: mean ATE, RMSE ATE, max ATE

*Relative Pose Error (RPE):*
- For configurable sub-sequence length `delta` (default 1.0m), compute relative pose error across all overlapping sub-sequences
- Publish: mean RPE translation, mean RPE rotation (degrees)

**Publications:**
```
/validation/ate_rmse      → std_msgs/Float64
/validation/rpe_trans     → std_msgs/Float64
/validation/rpe_rot       → std_msgs/Float64
/validation/report        → std_msgs/String (JSON, full metrics dump)
```

**Services:**
```
/validation/compute_now   → std_srvs/Trigger  (force immediate computation)
/validation/load_ground_truth  → (custom srv, takes file path string, loads reference .pcd or trajectory .txt)
/validation/set_baseline  → std_srvs/Trigger  (snapshot current metrics as pre-HBA baseline)
/validation/compare_to_baseline → std_srvs/Trigger (compute delta vs baseline, log pass/fail)
```

**Integration with `map_health_monitor`:**
- `map_health_monitor` calls `/validation/set_baseline` before triggering merge
- After VERIFYING phase, calls `/validation/compare_to_baseline`
- If ATE worsens by more than `ate_regression_threshold` (configurable, default 0.05m), this counts as a merge failure and triggers ROLLBACK

**Standalone mode:** The node must work without a ground truth source — in that case, skip ATE and compute only RPE (which is self-referential). This is the common case during real deployments.

---

### TASK 8 — PyQt5 GUI Extensions

Extend the existing `fastlio2_gui` to add a **Map Health Panel**. Do not break any existing GUI panels.

**New panel elements:**
- `QLabel` showing `map_health_monitor` current phase (MONITORING / HBA_RUNNING / MERGING / VERIFYING / ROLLBACK) with colour coding (green/yellow/orange/red)
- Mini `OccupancyGrid` renderer (using `QImage` + `QPainter`) showing the quality grid with colour ramp (green = healthy, red = degraded)
- Staleness grid toggle button (switch between quality and staleness overlay)
- HBA trigger count, last HBA timestamp, last merge outcome (COMMIT / ROLLBACK)
- ICP fitness trend plot — rolling 100-sample line chart using `pyqtgraph` or `matplotlib` embedded in Qt
- Manual override buttons:
  - `Force HBA Now` → publishes `HBATrigger` for current cell
  - `Force Merge Now` → publishes `MergeTrigger` with current policy
  - `Rollback Last Merge` → publishes rollback service call
- Merge policy dropdown (`TIMESTAMP_PRIORITY` / `DENSITY_PRIORITY` / `CONFIDENCE_WEIGHTED`) that sets the runtime ROS 2 parameter

---

## Package Structure

Create the following new packages inside the repo:

```
fastlio2_health/
├── CMakeLists.txt
├── package.xml
├── include/fastlio2_health/
│   ├── map_health_monitor.hpp
│   ├── pose_graph_partitioner.hpp
│   └── map_staleness_tracker.hpp
├── src/
│   ├── map_health_monitor.cpp
│   ├── pose_graph_partitioner.cpp
│   └── map_staleness_tracker.cpp
├── msg/
│   ├── MapHealthStatus.msg
│   ├── HBATrigger.msg
│   ├── HBACompletion.msg
│   ├── MergeTrigger.msg
│   └── MergeCompletion.msg
└── launch/
    └── health_monitor.launch.py

fastlio2_validation/
├── CMakeLists.txt
├── package.xml
├── include/fastlio2_validation/
│   └── map_validator.hpp
├── src/
│   └── map_validator.cpp
├── srv/
│   ├── LoadGroundTruth.srv
│   └── CompareToBaseline.srv
└── launch/
    └── validation.launch.py
```

---

## Bringup Integration

Update `fastlio2_bringup` to include a new launch file `full_system.launch.py` that starts:
1. FAST-LIO2 core (`laserMapping`)
2. FSM localiser
3. People filter
4. Loop closure detector
5. `map_health_monitor`
6. `map_validator` (standalone mode, no ground truth)
7. PyQt5 GUI

With a `use_health_monitor` launch argument (default `true`) to allow running without the health system for debugging.

Add the following parameters to the bringup YAML configs (one entry for MID-360, one for VLP-16):

```yaml
map_health_monitor:
  degradation_threshold: 0.4
  improvement_threshold: 0.05
  staleness_decay_s: 300.0
  staleness_hba_threshold: 0.8
  confidence_threshold: 0.7
  verify_scan_count: 30
  pre_merge_baseline_scans: 20
  partition_cell_size_m: 50.0
  partition_radius_cells: 1
  merge_policy: "TIMESTAMP_PRIORITY"

map_validator:
  delta_m: 1.0
  ate_regression_threshold: 0.05
```

---

## Testing Requirements

For each new package, provide a minimal launch-based integration test:

1. **`test_pipeline_wiring.py`** — Publishes synthetic ICP fitness degradation events and verifies that `map_health_monitor` transitions through TRIGGER_HBA → HBA_RUNNING → MERGING → VERIFYING within a timeout
2. **`test_degenerate_suppression.py`** — Publishes a degenerate flag and verifies that loop closure events are suppressed (no `LoopClosureEvent` published while flag is active)
3. **`test_spatial_partitioner.py`** — Unit test for `PoseGraphPartitioner`: insert 100 nodes across a 200m × 200m grid, verify that `getSubgraphNodes()` returns only nodes within the expected cells
4. **`test_ate_rpe.py`** — Feed a known trajectory with injected drift into `map_validator`, verify ATE and RPE outputs are within expected bounds

---

## Code Style Requirements

- C++17, `rclcpp` lifecycle nodes where appropriate
- All parameters declared with `declare_parameter()` + `get_parameter()`, not hardcoded
- All subscribers use appropriate QoS profiles (`SensorDataQoS` for pointclouds, `SystemDefaultsQoS` for control topics)
- No raw `new`/`delete` — use `std::shared_ptr` / `std::unique_ptr` throughout
- Log at appropriate levels: `RCLCPP_DEBUG` for per-scan events, `RCLCPP_INFO` for phase transitions, `RCLCPP_WARN` for threshold crossings, `RCLCPP_ERROR` for rollbacks
- Update the top-level `README.md` with a new **Architecture** section containing an ASCII diagram of the closed feedback loop and a table of all new topics/services/params

---

## Priority Order for Implementation

Implement in this order to ensure each phase is testable before the next begins:

1. **`map_health_monitor` skeleton** — node, subscriptions, publications, state machine (no spatial grid yet) → verify it builds and connects
2. **Automated HBA → merge → verify pipeline** — wire existing components via the state machine → verify end-to-end on a rosbag replay
3. **Loop closure event message + degenerate suppression** — small, high-value, verifiable in isolation
4. **Voxel-weighted merge policies** — extend existing merge, add runtime parameter
5. **Spatial pose graph partitioner** — add to `map_health_monitor`, unit test first
6. **Staleness-aware map sections** — extend `map_health_monitor` with `MapStalenessTracker`
7. **ATE/RPE validation node** — new package, integrate with health monitor
8. **PyQt5 GUI extensions** — last, depends on all topics being stable

---

## Final Note for Agent

The existing components in this repo are the work of a senior robotics engineer. Treat them as a stable API. Your job is to add the coordinator layer and integration wiring that connects them into a closed feedback loop. When in doubt about an existing component's interface, inspect its existing topic names and message types rather than assuming — the actual published topic names take precedence over anything in this prompt.
