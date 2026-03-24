# FASTLIO2_ROS2 — Extended Mapping & Localization Suite

> **Based on [liangheming/FASTLIO2_ROS2](https://github.com/liangheming/FASTLIO2_ROS2)**
> — a ROS2 port of [Fast-LIO2](https://github.com/hku-mars/FAST_LIO) by Xu *et al.*

This fork extends the upstream project with a full localization pipeline, a real-time people
filter, semantic point-cloud labelling, and a GUI-based launch system — forming a
production-ready system for indoor LiDAR-inertial mapping and localization with a
Livox MID360 sensor.

---

## Demo

<!-- Replace the link below with your YouTube / unlisted video URL -->
[![Demo Video](https://img.shields.io/badge/Watch%20Demo-YouTube-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=PLACEHOLDER)

> *Video coming soon — will show mapping, global re-localization, and people filtering in an indoor environment.*

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
9. [Credits](#credits)

---

## What's New in This Fork

The upstream [liangheming/FASTLIO2_ROS2](https://github.com/liangheming/FASTLIO2_ROS2) already provides
LiDAR-inertial odometry (`fastlio2`), global localization with NDT+ICP (`localizer`),
pose-graph optimization (`pgo`), hierarchical bundle adjustment (`hba`), and the service
interface definitions (`interface`).

This fork adds the following on top:

| Addition | Description |
|----------|-------------|
| **DBSCAN People Filter** | Real-time two-stage pipeline that removes human-shaped clusters from the live point cloud before it reaches the localizer, improving map-matching stability in crowded environments |
| **PCD Map Cleaner** | Offline CLI tool (`filter_map_pcd.py`) that applies the same geometry classifier to a saved PCD file, producing a people-free map for localization |
| **Semantic Segmentation** | RandLA-Net node that labels each incoming scan with SemanticKITTI classes (person, vehicle, ground, vegetation, …) and republishes per-class clouds for downstream use |
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
│          FASTLIO2  LIO Node  (C++)       │
│  IESKF  ·  ikd-Tree  ·  IMU Integrator   │
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
       │          │   NDT global search    │
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

---

## Package Overview

### Upstream (unchanged)

| Package | Language | Role |
|---------|----------|------|
| `fastlio2` | C++ | LiDAR-inertial odometry — IESKF + ikd-Tree SLAM core |
| `localizer` | C++ | Global localization (NDT+ICP) + continuous ICP tracking against a PCD map |
| `interface` | CMake | ROS2 service/message definitions |
| `pgo` | C++ | Pose-graph optimization |
| `hba` | C++ | Hierarchical bundle adjustment for map refinement |

### Added in This Fork

| Package | Language | Role |
|---------|----------|------|
| `fastlio2_bringup` | Python | Launch files + PyQt5 GUI control panel |
| `lidar_people_filter` | Python | Real-time DBSCAN people removal + offline PCD cleaner |
| `lidar_semantic` | Python/PyTorch | RandLA-Net semantic segmentation of incoming scans |

---

## Prerequisites

- **ROS2** Humble or later
- **Livox ROS Driver 2** — built in a separate workspace (e.g. `~/livox_ws`)
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
cd ~/ros2_ws/src
git clone https://github.com/knnurl/FASTLIO2_ROS2.git

# 3. Build
cd ~/ros2_ws
colcon build --symlink-install --packages-select \
    interface fastlio2 localizer fastlio2_bringup lidar_people_filter lidar_semantic

# 4. Source the workspace
source install/setup.bash
```

### Download semantic weights (optional)

```bash
cd src/FASTLIO2_ROS2/lidar_semantic
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
- Call services (Save Map, Force Re-Localize, Check Status) with one click

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

**Trigger manual re-localization at any time** (e.g. after the robot has been moved):

```bash
ros2 service call /localizer/global_relocalize interface/srv/GlobalRelocalize \
    "{pcd_path: '', force: true}"
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
| `/fastlio2/save_map` | `interface/srv/SaveMaps` | Write current ikd-tree map to a PCD file |
| `/localizer/global_relocalize` | `interface/srv/GlobalRelocalize` | Trigger NDT+ICP global search; returns pose + fitness score |
| `/localizer/is_valid` | `interface/srv/IsValid` | Query whether the localizer is tracking confidently |
| `/localizer/start_mapping` | `interface/srv/StartMapping` | Begin auxiliary map update |
| `/localizer/stop_mapping` | `interface/srv/StopMapping` | End map update and merge result |

---

## Configuration Reference

### `fastlio2/config/lio.yaml`

Core LIO parameters: feature thresholds, voxel filter sizes, ikd-tree parameters, and IMU noise values. Adjust `filter_size_surf` and `filter_size_map` for the speed/density trade-off.

### `localizer/config/localizer.yaml`

Global search resolution (NDT voxel size, angular sweep step) and ICP tracking thresholds (max correspondence distance, convergence tolerance, fitness score cutoff for LOST→SEARCHING transition).

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
