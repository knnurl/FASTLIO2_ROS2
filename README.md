# FASTLIO2_ROS2 — Extended Mapping & Localization Suite

> **Experimental snapshot (April 2026), not maintained.** This branch holds an untested prototype of a closed-loop map-health layer. It has known problems, including an inverted rollback check; see [docs/health-prototype/STATUS.md](docs/health-prototype/STATUS.md). The maintained code is on `main`.

> **Based on [liangheming/FASTLIO2_ROS2](https://github.com/liangheming/FASTLIO2_ROS2)**
> — a ROS2 port of [Fast-LIO2](https://github.com/hku-mars/FAST_LIO) by Xu *et al.*

This fork extends the upstream project with a full localization pipeline, a real-time people
filter, semantic point-cloud labelling, and a GUI-based launch system — forming a
production-ready system for indoor LiDAR-inertial mapping and localization with a
Livox MID360 sensor.

---

## Demo

<div align="center">
  <a href="https://youtu.be/Ip-UA0tPhvY">
    <img src="https://i.ytimg.com/vi/Ip-UA0tPhvY/maxresdefault.jpg" width="800" alt="FASTLIO2 Demo Video">
  </a>
</div>
<br>

> *This demo showcases the system's core capabilities using the GUI Control Panel, including real-time point-cloud mapping, automatic global re-localization recovering from tracking failures (via global search), and the ability to seamlessly update and merge an existing map while exploring new areas.*

---

## Table of Contents

1. [Demo](#demo)
2. [What's New in This Fork](#whats-new-in-this-fork)
3. [System Architecture](#system-architecture)
4. [Package Overview](#package-overview)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [Usage](#usage)
   - [GUI Control Panel](#gui-control-panel)
   - [Mapping](#mapping)
   - [Localization](#localization)
   - [People Filter](#people-filter)
   - [Semantic Segmentation](#semantic-segmentation)
8. [ROS2 Topics & Services](#ros2-topics--services)
9. [Configuration Reference](#configuration-reference)
10. [Credits](#credits)

---

## What's New in This Fork

The upstream [liangheming/FASTLIO2_ROS2](https://github.com/liangheming/FASTLIO2_ROS2) already provides
LiDAR-inertial odometry (`fastlio2`), global localization with NDT+ICP (`localizer`),
pose-graph optimization (`pgo`), hierarchical bundle adjustment (`hba`), and the service
interface definitions (`interface`).

This fork adds the following on top:

### Algorithmic Contributions

**Extended Localizer** (modifications to `localizer/`)

The upstream localizer supported manual ICP-only localization (pose hint required).
The following were added on top — the original ICP path is explicitly left unchanged
(`// ICP mode (original logic, unchanged)` comment in `localizer_node.cpp`):

- **Global mode state machine** — a four-state FSM (IDLE → SEARCHING → TRACKING → LOST)
  that drives the full localization lifecycle without any user-provided pose hint.
- **Asynchronous global search** — the NDT+ICP grid search runs in a `std::async` thread
  with an odometry snapshot taken at launch time, so the map↔local offset is anchored
  to the correct moment even after a multi-second search.
- **Pose persistence with map-staleness detection** — the current map↔body transform is
  saved to YAML on disk at a configurable rate. On restart, `loadPose()` cross-checks the
  saved pose against the map file's modification time and silently discards it if the map
  has changed, falling back to a full global search.
- **ICP jump guard** — after each ICP alignment the result translation is compared against
  the odometry prediction; results exceeding `icp_max_jump_m` are rejected as wrong local
  minima rather than corrupting the offset.
- **Progressive ICP recovery** — at the half-failure threshold, correspondence distances
  are widened by `recovery_corr_scale` to give ICP a wider reach before a full global
  search is triggered. Distances are restored automatically once tracking recovers.
- **LIO dead-reckoning fallback** — if a global search fails *after* the robot previously
  had a good pose (i.e. entered unmapped territory), the node holds LIO dead-reckoning
  instead of entering an IDLE→SEARCH→FAIL spin that would corrupt the position estimate.
- **Degenerate scan guard** — scans below `min_cloud_points` are skipped entirely,
  preventing wasted ICP computation on empty or nearly-empty returns.
- **Hot-reload of merged map** — after `stop_mapping` saves a merged PCD, the node
  reloads it into the ICP localizer on the next timer tick without requiring a restart.
- **Two-stage adaptive ICP** (`icp_localizer.cpp`) — rough and refine stages use separate
  voxel resolutions; if the rough fitness is marginal, the refine correspondence distance
  is automatically widened via `recovery_corr_scale` to avoid divergence from a
  slightly-off starting point.
- **Auxiliary map update services** (`start_mapping` / `stop_mapping`) — while localizing,
  the node can accumulate incoming scans into a secondary cloud. On `stop_mapping`, the
  cloud is voxel-filtered, written to PCD, and the ICP localizer hot-reloads it.

**Map Save Service** (addition to `fastlio2/`)

The upstream `fastlio2` node had no `/fastlio2/save_map` service implementation despite
the interface definition existing. The service handler is now implemented: it extracts all
valid points from the live ikd-Tree, writes them as a binary PCD file to the requested
path (creating parent directories as needed), and returns a success flag and point count.

**Adaptive DBSCAN People Filter** (`lidar_people_filter`)

A real-time human removal pipeline designed specifically for Livox MID360 point clouds
in indoor environments:

- **Range-adaptive ε** — DBSCAN neighbourhood radius scales with median scan range
  (1.2–1.5× multiplier beyond 3 m), compensating for the sensor's range-dependent
  point density so far-range humans cluster as reliably as close-range ones.
- **Voxel pre-downsampling** — normalises density before clustering so the geometry
  classifier sees consistent cluster shapes regardless of range or scan overlap.
- **Temporal consistency tracker** (`RepeatedHumanTracker`) — a sliding-window
  voxel-hit accumulator that tracks classified clusters over a configurable frame window
  (default 12 frames). Voxels present in ≥ 70% of frames are flagged as static
  structures and kept; voxels that appear/disappear are removed as dynamic people.
  This catches stationary people that geometry alone cannot distinguish from furniture.
- **Multi-criterion geometry classifier** — height, footprint diameter, aspect ratio,
  point count, and an explicit single-axis width guard to prevent wall segments from
  being misclassified as humans.
- **Offline map cleaner** (`filter_map_pcd.py`) — applies the same classifier to a
  saved PCD file, with a spatial remapping step that recovers full-resolution points
  from the downsampled clustering result.

**RandLA-Net Semantic Segmentation** (`lidar_semantic`)

Integration of RandLA-Net (CVPR 2020) for per-scan SemanticKITTI labelling:

- Decoupled inference via a background worker thread — the ROS subscriber only
  updates a pending-message slot; the worker drains it asynchronously, keeping
  the ROS graph responsive under GPU load.
- Multi-scale KNN graph and random downsampling indices are precomputed in NumPy
  before each forward pass, avoiding redundant GPU↔CPU transfers.
- Class-conditional output streams: person cloud, ground cloud, and a full labelled
  cloud are published separately for downstream consumption.

### Infrastructure Contributions

| Addition | Description |
|----------|-------------|
| **Launch Orchestration** | Modular launch files covering every combination of live/rosbag, mapping/localization, with/without people filter and RViz |
| **PyQt5 GUI** | Dark-themed control panel for launching stacks, toggling the Livox driver, triggering services, and monitoring output — no terminal required |

---

## System Architecture

```
┌──────────────────────────────────────────┐
│         Livox MID360 Driver              │
│   livox_ros_driver2_node  @10 Hz         │
└───────────────┬──────────────────────────┘
                │ /livox/lidar  /livox/imu
                ▼
┌──────────────────────────────────────────┐
│          FASTLIO2  LIO Node  (C++)        │
│  IESKF  ·  ikd-Tree  ·  IMU Integrator  │
│                                          │
│  /fastlio2/body_cloud   (body frame)     │
│  /fastlio2/world_cloud  (world frame)    │
│  /fastlio2/lio_odom                      │
│  /fastlio2/accumulated_map               │
└──────┬────────────────┬──────────────────┘
       │                │
       │         ┌──────▼──────────────────┐
       │         │  DBSCAN People Filter   │
       │         │  (optional, live)       │
       │         │  /fastlio2/body_cloud   │
       │         │    → body_cloud_filtered│
       │         └──────┬──────────────────┘
       │                │
  MAPPING           LOCALIZATION
  MODE                  │
       │          ┌─────▼──────────────────┐
       │          │   ICP Localizer (C++)  │
       │          │   Global search (force)│
       │          │   → continuous ICP     │
       │          │   → TF: map→base_link  │
       │          └────────────────────────┘
       │
  /fastlio2/save_map ──▶ map.pcd
                                ┌──────────────────────────┐
  /fastlio2/body_cloud ────────▶│  RandLA-Net  (optional)  │
                                │  SemanticKITTI · 19 cls  │
                                │  /semantic/person_cloud  │
                                │  /semantic/ground_cloud  │
                                └──────────────────────────┘
```

**TF tree**

```
map
 └── camera_init  (published by localizer when tracking)
      └── body    (published by LIO node)
```

### Closed Feedback Loop Architecture

The `fastlio2_health` package connects the existing components into a self-healing
closed-loop system. The `map_health_monitor` node is the central coordinator.

```
                        ┌──────────────────────────────────┐
                        │       map_health_monitor         │
                        │  ┌────────────────────────────┐  │
                        │  │   Pipeline State Machine   │  │
                        │  │  MONITORING → TRIGGER_HBA  │  │
                        │  │  → HBA_RUNNING → MERGING   │  │
                        │  │  → VERIFYING → COMMIT      │  │
                        │  │        or → ROLLBACK       │  │
                        │  └────────────────────────────┘  │
                        │                                  │
                        │  ┌──────────┐ ┌──────────────┐   │
                        │  │ Spatial  │ │  Staleness   │   │
                        │  │ Quality  │ │  Tracker     │   │
                        │  │  Grid    │ │              │   │
                        │  └──────────┘ └──────────────┘   │
                        │                                  │
                        │  ┌────────────────────────────┐  │
                        │  │  PoseGraphPartitioner      │  │
                        │  │  (spatial subgraph select) │  │
                        │  └────────────────────────────┘  │
                        └──────┬───────────┬───────────────┘
                               │           │
             /hba/trigger      │           │  /map_merge/trigger
                    ┌──────────▼──┐   ┌────▼──────────────────┐
                    │  HBA Node   │   │  Voxel Merge Node     │
                    │  (existing) │   │  TIMESTAMP_PRIORITY   │
                    │             │   │  DENSITY_PRIORITY     │
                    └──────┬──────┘   │  CONFIDENCE_WEIGHTED  │
                           │          └────┬──────────────────┘
             /hba/completion               │  /map_merge/completion
                           │               │
                    ┌──────▼───────────────▼──────────┐
                    │        map_health_monitor       │
                    │  (evaluates post-merge quality) │
                    │  → COMMIT or ROLLBACK           │
                    └──────┬──────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   map_validator         │
              │   ATE/RPE metrics       │
              │   /validation/set_baseline
              │   /validation/compare   │
              └─────────────────────────┘

  ───── Data Sources ─────

  /fastlio2/lio_odom ──────► robot position → quality cell lookup
  /fsm/icp_fitness ────────► quality grid update, degradation check
  /fsm/state ──────────────► FSM state awareness
  /loop_closure/event ─────► accepted closures trigger HBA
  /degenerate_guard/suppress_loop_closure ──► suppresses false closures

  ───── Outputs ─────

  /map_health/quality_grid ──► OccupancyGrid (RViz: green=ok, red=bad)
  /map_health/staleness_grid ► OccupancyGrid (RViz: stale regions)
  /map_health/status ────────► MapHealthStatus (global score + phase)
  /map_health/gui_state ─────► JSON blob for PyQt5 GUI panel
```

### New Topics (Health System)

| Topic | Type | Publisher | Description |
|-------|------|-----------|-------------|
| `/map_health/quality_grid` | `nav_msgs/OccupancyGrid` | map_health_monitor | Spatial map quality heatmap |
| `/map_health/staleness_grid` | `nav_msgs/OccupancyGrid` | map_health_monitor | Spatial staleness heatmap |
| `/map_health/status` | `fastlio2_health/MapHealthStatus` | map_health_monitor | Global health score + phase |
| `/map_health/gui_state` | `std_msgs/String` | map_health_monitor | JSON state for GUI |
| `/hba/trigger` | `fastlio2_health/HBATrigger` | map_health_monitor | Triggers HBA on subgraph |
| `/hba/completion` | `fastlio2_health/HBACompletion` | hba_node | HBA result + corrected PCD |
| `/map_merge/trigger` | `fastlio2_health/MergeTrigger` | map_health_monitor | Triggers map merge |
| `/map_merge/completion` | `fastlio2_health/MergeCompletion` | voxel_merge_node | Merge result |
| `/map_merge/rollback` | `std_msgs/Bool` | map_health_monitor | Rollback last merge |
| `/loop_closure/event` | `fastlio2_health/LoopClosureEvent` | pgo_node | Loop closure detections |
| `/degenerate_guard/suppress_loop_closure` | `std_msgs/Bool` | degenerate_guard | Suppress loop closure |
| `/validation/ate_rmse` | `std_msgs/Float64` | map_validator | ATE RMSE metric |
| `/validation/rpe_trans` | `std_msgs/Float64` | map_validator | RPE translation metric |
| `/validation/rpe_rot` | `std_msgs/Float64` | map_validator | RPE rotation metric |
| `/validation/report` | `std_msgs/String` | map_validator | Full JSON metrics report |

### New Services (Health System)

| Service | Interface | Description |
|---------|-----------|-------------|
| `/validation/compute_now` | `std_srvs/Trigger` | Force immediate metric computation |
| `/validation/set_baseline` | `std_srvs/Trigger` | Snapshot metrics as pre-merge baseline |
| `/validation/compare_to_baseline` | `fastlio2_validation/CompareToBaseline` | Compare current vs baseline (PASS/FAIL) |
| `/validation/load_ground_truth` | `fastlio2_validation/LoadGroundTruth` | Load reference trajectory |

### New Parameters (Health System)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `degradation_threshold` | `0.4` | Combined degradation score that triggers HBA |
| `improvement_threshold` | `0.05` | Minimum fitness improvement to accept merge |
| `staleness_decay_s` | `300.0` | Seconds until a cell becomes fully stale |
| `staleness_hba_threshold` | `0.8` | Staleness score that elevates HBA priority |
| `confidence_threshold` | `0.7` | ICP fitness above which localisation is confident |
| `verify_scan_count` | `30` | Scans to collect during post-merge verification |
| `pre_merge_baseline_scans` | `20` | Scans for pre-merge fitness baseline |
| `partition_cell_size_m` | `50.0` | Spatial partitioner cell size |
| `partition_radius_cells` | `1` | Adjacent cells to include in HBA subgraph |
| `merge_policy` | `TIMESTAMP_PRIORITY` | Default merge strategy |
| `delta_m` | `1.0` | RPE sub-sequence length |
| `ate_regression_threshold` | `0.05` | ATE worsening that triggers rollback |

---

## Package Overview

### Upstream (unchanged)

| Package | Language | Role |
|---------|----------|------|
| `fastlio2` | C++ | LiDAR-inertial odometry — IESKF + ikd-Tree SLAM core + `/fastlio2/save_map` service |
| `localizer` | C++ | Global localization (ICP) + continuous ICP tracking + auxiliary map update |
| `interface` | CMake | ROS2 service/message definitions |
| `pgo` | C++ | Pose-graph optimization |
| `hba` | C++ | Hierarchical bundle adjustment for map refinement |

### Added in This Fork

| Package | Language | Role |
|---------|----------|------|
| `fastlio2_bringup` | Python | Launch files + PyQt5 GUI control panel |
| `lidar_people_filter` | Python | Real-time DBSCAN people removal + offline PCD cleaner |
| `lidar_semantic` | Python/PyTorch | RandLA-Net semantic segmentation of incoming scans |
| `fastlio2_health` | C++ | Map health monitor, voxel merge node, spatial partitioner, staleness tracker |
| `fastlio2_validation` | C++ | ATE/RPE trajectory validation node |

---

## Prerequisites

- **ROS2** Humble or later
- **Livox ROS Driver 2** — built in a separate workspace (e.g. `~/livox_ws`)
- **GTSAM** — required by `hba` and `pgo` for pose-graph optimization:
  ```bash
  sudo apt install libgtsam-dev
  ```
  If not available via apt, add the borglab PPA first:
  ```bash
  sudo add-apt-repository ppa:borglab/gtsam-release-4.1
  sudo apt update && sudo apt install libgtsam-dev libgtsam-unstable-dev
  ```
- **Python ≥ 3.8** with `PyQt5`, `numpy`, `open3d`, `scikit-learn`
- **PyTorch ≥ 1.12** with CUDA (for `lidar_semantic` — CPU fallback available)
- A **Livox MID360** or compatible sensor (for live data mode)

```bash
pip install PyQt5 numpy open3d scikit-learn torch torchvision
```

---

## Installation

```bash
# 1. Source Livox driver workspace
source ~/livox_ws/install/setup.bash

# 2. Clone this repo into your ROS2 workspace
mkdir -p ~/fastlio3_ws/src
cd ~/fastlio3_ws/src
git clone <repo-url> .

# 3. Build
cd ~/fastlio3_ws
colcon build --symlink-install --allow-overriding interface

# 4. Source the workspace
source install/setup.bash
```

> **Note:** `--allow-overriding interface` is required because `interface` is also present in
> the upstream workspaces (`fastlio_ws`, `fastlio2_ws`). Without it colcon will refuse to build.

### Download semantic weights (optional)

```bash
cd src/lidar_semantic
python3 scripts/download_weights.py
```

---

## Usage

### GUI Control Panel

The easiest way to operate the full system:

```bash
ros2 run fastlio2_bringup fastlio2_gui.py
```

The panel lets you:

- Switch between **Mapping** and **Localization** modes
- Toggle **Live data** to launch the Livox driver, or leave it unchecked to replay a rosbag
- Browse to a `.pcd` map file
- Start / stop the stack and monitor live log output
- Call services (Save Map, Force Re-Localize, Start/Stop Map Update, Check Status) with one click

### Mapping

**With live sensor (GUI or terminal):**

```bash
ros2 launch fastlio2_bringup mapping_full.launch.py
```

**Rosbag replay (driver-less):**

```bash
ros2 launch fastlio2_bringup mapping.launch.py
# In another terminal:
ros2 bag play <your_bag>
```

**Save the map once mapping is complete:**

```bash
ros2 service call /fastlio2/save_map interface/srv/SaveMaps \
    "{file_path: '$HOME/maps/map.pcd', save_patches: false}"
```

**Optional — remove people from the saved map:**

```bash
python3 src/lidar_people_filter/scripts/filter_map_pcd.py \
    ~/maps/map.pcd --output ~/maps/map_clean.pcd
```

### Localization

**With live sensor:**

```bash
ros2 launch fastlio2_bringup localization_full.launch.py \
    map_path:=$HOME/maps/map_clean.pcd
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `map_path` | `maps/map.pcd` | Path to pre-saved PCD map |
| `use_people_filter` | `true` | Enable real-time DBSCAN filter |
| `force_global_search` | `true` | Ignore saved pose; always run full NDT+ICP search |
| `launch_rviz` | `true` | Start RViz2 (set `false` when using the GUI) |

**Rosbag replay:**

```bash
ros2 launch fastlio2_bringup localization.launch.py \
    map_path:=$HOME/maps/map_clean.pcd
```

**Trigger global re-localization at any time** (e.g. after the robot has been moved):

```bash
ros2 service call /localizer/global_relocalize interface/srv/GlobalRelocalize \
    "{pcd_path: '', force: true}"
```

Pass `pcd_path` to also reload the map from a new file before searching.

**Update the map while localizing** (append new areas without stopping):

```bash
# Start accumulating new scans
ros2 service call /localizer/start_mapping interface/srv/StartMapping \
    "{output_path: '$HOME/maps/map_updated.pcd', voxel_size: 0.05}"

# ... drive the robot through the new area ...

# Stop, save, and hot-reload the merged map
ros2 service call /localizer/stop_mapping interface/srv/StopMapping \
    "{output_path: '$HOME/maps/map_updated.pcd'}"
```

### People Filter

Launched automatically by `localization_full.launch.py` when `use_people_filter:=true`.
To run it standalone (e.g., alongside a custom stack):

```bash
ros2 launch lidar_people_filter people_filter.launch.py
```

The filter subscribes to `/fastlio2/body_cloud` and publishes the cleaned cloud on
`/fastlio2/body_cloud_filtered`.  Tune `config/dbscan_filter.yaml` to match your
environment's crowd density and scan characteristics.

### Semantic Segmentation

```bash
ros2 launch lidar_semantic semantic.launch.py \
    input_topic:=/fastlio2/body_cloud_filtered \
    device:=cuda
```

Published topics:

| Topic | Content |
|-------|---------|
| `/semantic/labelled_cloud` | Full scan — intensity = class index, RGB packed as float32 |
| `/semantic/person_cloud` | Person + bicyclist + motorcyclist points only |
| `/semantic/ground_cloud` | Road + parking + sidewalk + terrain points |
| `/semantic/markers` | RViz sphere markers, one per detected cluster |

---

## ROS2 Topics & Services

### Topics

| Topic | Type | Publisher | Description |
|-------|------|-----------|-------------|
| `/fastlio2/body_cloud` | `sensor_msgs/PointCloud2` | fastlio2 | Current scan in body (LiDAR) frame |
| `/fastlio2/world_cloud` | `sensor_msgs/PointCloud2` | fastlio2 | Current scan in world frame |
| `/fastlio2/lio_odom` | `nav_msgs/Odometry` | fastlio2 | LIO odometry at scan rate |
| `/fastlio2/lio_path` | `nav_msgs/Path` | fastlio2 | Full trajectory history |
| `/fastlio2/accumulated_map` | `sensor_msgs/PointCloud2` | fastlio2 | Periodically downsampled map cloud |
| `/fastlio2/body_cloud_filtered` | `sensor_msgs/PointCloud2` | people_filter | People-removed body cloud |
| `/localizer/map_cloud` | `sensor_msgs/PointCloud2` | localizer | Pre-saved map (for RViz) |

### Services

| Service | Interface | Description |
|---------|-----------|-------------|
| `/fastlio2/save_map` | `interface/srv/SaveMaps` | Extract ikd-Tree map and write to a PCD file |
| `/localizer/global_relocalize` | `interface/srv/GlobalRelocalize` | Trigger ICP global search; returns pose + fitness score |
| `/localizer/is_valid` | `interface/srv/IsValid` | Query whether the localizer is tracking confidently |
| `/localizer/start_mapping` | `interface/srv/StartMapping` | Begin accumulating scans for auxiliary map update |
| `/localizer/stop_mapping` | `interface/srv/StopMapping` | Save accumulated scans to PCD and hot-reload the localizer |
| `/localizer/relocalize` | `interface/srv/Relocalize` | ICP relocalization with explicit pose hint (x, y, z, yaw, pitch, roll) |

#### GlobalRelocalize request/response

```
# Request
string pcd_path   # optional: reload map from this path before searching (empty = use loaded map)
bool   force      # reserved for future use; pass true

# Response
bool    success
string  message
float32 fitness_score   # lower is better (mean squared point-to-point distance)
float32 x               # estimated position (m)
float32 y
float32 z
float32 yaw_deg         # estimated heading (degrees)
```

---

## Configuration Reference

### `fastlio2/config/lio.yaml`

Core LIO parameters: feature thresholds, voxel filter sizes, ikd-tree parameters, and IMU noise values. Adjust `filter_size_surf` and `filter_size_map` for the speed/density trade-off.

### `localizer/config/localizer.yaml`

ICP tracking thresholds (voxel resolutions, max iterations, fitness score cutoffs).

| Key | Default | Description |
|-----|---------|-------------|
| `rough_scan_resolution` | `0.25` | Voxel size for rough-stage scan downsampling (m) |
| `rough_map_resolution` | `0.25` | Voxel size for rough-stage map downsampling (m) |
| `rough_max_iteration` | `5` | Max ICP iterations in rough stage |
| `rough_score_thresh` | `0.2` | Fitness threshold to accept rough alignment |
| `refine_scan_resolution` | `0.1` | Voxel size for refine-stage scan downsampling (m) |
| `refine_map_resolution` | `0.1` | Voxel size for refine-stage map downsampling (m) |
| `refine_max_iteration` | `10` | Max ICP iterations in refine stage |
| `refine_score_thresh` | `0.1` | Fitness threshold to accept refined alignment |
| `update_hz` | `1.0` | Rate at which the map→local TF is updated (Hz) |

### `lidar_people_filter/config/dbscan_filter.yaml`

```yaml
# DBSCAN clustering
voxel_size: 0.10        # pre-downsample resolution (m)
eps: 0.40               # neighbourhood radius
min_samples: 3          # minimum points per cluster

# Geometry-based human classifier
human_min_height: 0.8   # m — minimum cluster height
human_max_height: 2.2   # m — maximum cluster height
human_min_footprint: 0.1
human_max_footprint: 1.0
aspect_ratio_min: 1.2   # height/width — rejects flat objects

# Temporal consistency
temporal_window: 12     # frames to track voxel history
static_threshold: 0.70  # fraction of frames → static, keep
```

### `lidar_semantic/config/semantic.yaml`

Model path, inference device (`cuda` / `cpu`), number of points per inference (16 384),
and per-class confidence threshold.

---

## Credits

- **FAST-LIO2** — Xu *et al.*, HKU-MARS Lab.
  [Paper](https://arxiv.org/abs/2107.06829) · [Code](https://github.com/hku-mars/FAST_LIO)
- **FASTLIO2_ROS2** — ROS2 port by
  [liangheming](https://github.com/liangheming/FASTLIO2_ROS2)
- **RandLA-Net** — Hu *et al.*, CVPR 2020.
  [Paper](https://arxiv.org/abs/1911.11236) · pretrained weights from
  [tsunghan-parent/randlanet-pytorch](https://github.com/tsunghan-parent/randlanet-pytorch)
- **Livox ROS Driver 2** — [Livox-SDK/livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2)
