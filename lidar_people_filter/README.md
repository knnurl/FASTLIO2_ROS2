# lidar_people_filter

**ROS2 package** — real-time detection and removal of people from 3D LiDAR scans for clean 3D mapping.

---

## Architecture

```
LiDAR (/velodyne_points)
        │
        ▼
┌──────────────────────┐
│  people_filter_node  │  geometry-based human detection + removal
└──────────────────────┘
        │ /points_no_people
        ▼
┌────────────────────────────┐
│ background_subtractor_node │  optional: static-scene voxel subtraction
└────────────────────────────┘
        │ /points_static
        ▼
   Mapper (e.g. Cartographer, RTAB-Map, OctoMap)
```

### Pipeline per scan

1. **Voxel downsample** — reduce point density (default 5 cm grid)
2. **Ground removal** — RANSAC plane fit strips the floor
3. **Clustering** — DBSCAN (preferred) or Euclidean clustering groups non-ground points
4. **Human classification** — per-cluster bounding-box geometry rules:
   - Height: 1.2 – 2.2 m
   - Width / depth: 0.2 – 1.2 m
   - Roughly circular horizontal cross-section
   - Point count within expected range
5. **Point removal** — all raw scan points inside inflated human boxes are discarded
6. **Publish** — clean cloud forwarded to mapping stack

---

## Installation

### Dependencies

```bash
# ROS2 (Humble / Iron / Jazzy)
sudo apt install ros-$ROS_DISTRO-sensor-msgs \
                 ros-$ROS_DISTRO-visualization-msgs \
                 ros-$ROS_DISTRO-std-srvs

# Python
pip install scikit-learn numpy   # sklearn optional but recommended
```

### Build

```bash
cd ~/ros2_ws/src
git clone <this-repo> lidar_people_filter
cd ~/ros2_ws
colcon build --packages-select lidar_people_filter
source install/setup.bash
```

---

## Usage

### Quick start

```bash
ros2 launch lidar_people_filter people_filter.launch.py \
    input_topic:=/velodyne_points \
    use_rviz:=true
```

### With Velodyne driver

```bash
# Terminal 1 – sensor driver
ros2 launch velodyne_driver velodyne_driver_node.launch.py

# Terminal 2 – people filter
ros2 launch lidar_people_filter people_filter.launch.py

# Terminal 3 – mapper (e.g. OctoMap)
ros2 run octomap_server octomap_server_node \
    --ros-args -r cloud_in:=/points_static
```

### With RTAB-Map

```bash
ros2 launch lidar_people_filter people_filter.launch.py \
    static_topic:=/rtabmap/cloud_map
```

---

## Background Subtraction (optional)

The `background_subtractor_node` builds a static-scene voxel map and removes
any foreground (dynamic) points not present in the background.

```bash
# 1. Launch with environment clear of people
ros2 launch lidar_people_filter people_filter.launch.py auto_calibrate:=true

# OR: trigger calibration manually after launch
ros2 service call /background_subtractor_node/calibrate std_srvs/srv/Trigger {}

# Reset background model
ros2 service call /background_subtractor_node/reset std_srvs/srv/Trigger {}
```

---

## Parameters

Edit `config/people_filter.yaml` or override on command line.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `voxel_size` | 0.05 | Downsample grid size (m) |
| `cluster_tolerance` | 0.5 | DBSCAN ε / cluster radius (m) |
| `use_dbscan` | true | Use DBSCAN (sklearn) vs simple Euclidean |
| `human_min_height` | 1.2 | Minimum human height (m) |
| `human_max_height` | 2.2 | Maximum human height (m) |
| `human_min_width` | 0.2 | Minimum width (m) |
| `human_max_width` | 1.2 | Maximum width (m) |
| `removal_inflation` | 0.2 | Extra margin around removed boxes (m) |
| `publish_markers` | true | Publish RViz bounding boxes |

### Tuning tips

- **Outdoors / high density** → increase `voxel_size` (0.10–0.15 m) for speed
- **Slow walker false-negatives** → widen `human_max_width` slightly
- **Tree trunks misclassified** → tighten `human_max_horizontal_aspect` to 2.0
- **Children missed** → lower `human_min_height` to 0.9 m
- **Wheelchairs / scooters** → expand `human_max_width` to 1.5 m

---

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `input_topic` | `PointCloud2` | Raw LiDAR input |
| `output_topic` | `PointCloud2` | Cloud with people removed |
| `marker_topic` | `MarkerArray` | RViz detection bounding boxes |
| `/points_static` | `PointCloud2` | After background subtraction |

---

## Extending

### Swap to ML-based detector

Replace `classify_human()` in `people_filter_node.py` with a call to a
PointNet / PointPillars model via `torch`:

```python
def classify_human(feat, params):
    tensor = cluster_to_tensor(feat)          # your preprocessing
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0))
    return logits.argmax().item() == HUMAN_CLASS
```

### Pipe into LIO-SAM / FAST-LIO

```bash
ros2 launch lidar_people_filter people_filter.launch.py \
    static_topic:=/lidar_points      # LIO-SAM default input topic
```

---

## License

MIT
