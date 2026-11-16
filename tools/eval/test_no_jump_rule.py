"""Roadmap B2: the odometry that goes into the flight controller never jumps.

The rule holds by construction, and these tests keep it that way:
- lio_node takes IMU and LiDAR and nothing else: no topic, service or client
  through which another node could move its state;
- pgo, hba and the localizer publish corrections only as a TF whose parent is
  the map frame, and never publish odometry.
Plus the measurement used on replays (metrics.odometry_steps).
"""
import pathlib
import re

import numpy as np

from metrics import odometry_steps

ROOT = pathlib.Path(__file__).resolve().parents[2]


def src(rel):
    return (ROOT / rel).read_text()


def test_lio_node_has_no_input_that_could_move_its_state():
    text = src('fastlio2/src/lio_node.cpp')
    subs = re.findall(r'create_subscription<([\w:]+)>', text)
    assert sorted(subs) == ['livox_ros_driver2::msg::CustomMsg', 'sensor_msgs::msg::Imu',
                            'sensor_msgs::msg::PointCloud2'], subs
    services = re.findall(r'create_service<([\w:]+)>', text)
    assert services == ['interface::srv::SaveMaps'], services
    assert 'create_client' not in text
    assert 'TransformListener' not in text and 'lookupTransform' not in text


def test_filter_state_is_written_only_by_the_filter():
    # No assignment to the pose outside the filter and the IMU initialisation.
    for rel in ('fastlio2/src/lio_node.cpp', 'fastlio2/src/map_builder/lidar_processor.cpp',
                'fastlio2/src/map_builder/map_builder.cpp'):
        assert not re.search(r'x\(\)\.(t_wi|r_wi)\s*=[^=]', src(rel)), rel


def test_map_side_nodes_correct_only_the_map_frame():
    for rel in ('pgo/src/pgo_node.cpp', 'localizer/src/localizer_node.cpp'):
        text = src(rel)
        assert 'create_publisher<nav_msgs::msg::Odometry>' not in text, rel
        assert 'create_client' not in text, rel
        parents = re.findall(r'header\.frame_id\s*=\s*([\w.]+);', text)
        tf_parents = [p for p in parents if 'frame' in p]
        assert tf_parents and all(p.endswith('map_frame') for p in tf_parents), (rel, parents)
    assert 'create_publisher<nav_msgs::msg::Odometry>' not in src('hba/src/hba_node.cpp')


def test_odometry_steps_metric():
    t = np.arange(0, 5, 0.1)
    p = np.stack([t * 1.0, np.zeros_like(t), np.zeros_like(t)], axis=1)   # 1 m/s
    clean = odometry_steps(t, p)
    assert clean['jumps'] == 0 and abs(clean['max_step_m'] - 0.1) < 1e-9
    p[30:, 0] += 2.0                                                      # a 2 m correction
    bad = odometry_steps(t, p)
    assert bad['jumps'] == 1 and bad['max_step_m'] > 2.0
    assert odometry_steps([0.0], [[0, 0, 0]])['jumps'] == 0
