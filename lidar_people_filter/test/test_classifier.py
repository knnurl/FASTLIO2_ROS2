"""Unit tests for the DBSCAN people filter's pure helpers.

The classifier tests use the tuned values in config/dbscan_filter.yaml, so they
document what the shipped configuration accepts as a person.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml
from std_msgs.msg import Header

from lidar_people_filter.dbscan_filter_node import (
    RepeatedHumanTracker, is_human_cluster, pc2_to_xyz, voxel_downsample, xyz_to_pc2)

CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'dbscan_filter.yaml'


@pytest.fixture(scope='module')
def params():
    return yaml.safe_load(CONFIG.read_text())['dbscan_filter_node']['ros__parameters']


def box(dx, dy, dz, n=60, origin=(2.0, 1.0, 0.0), seed=0):
    """n points filling a dx*dy*dz box, with two opposite corners included
    so the extents are exact."""
    rng = np.random.default_rng(seed)
    o = np.asarray(origin, dtype=np.float32)
    size = np.array([dx, dy, dz], dtype=np.float32)
    pts = o + rng.random((n, 3), dtype=np.float32) * size
    pts[0], pts[1] = o, o + size
    return pts


class TestIsHumanCluster:

    def test_standing_person(self, params):
        assert is_human_cluster(box(0.4, 0.3, 1.7), params)

    def test_partial_body_at_range(self, params):
        # 0.8 m minimum height is tuned for partial returns at range
        assert is_human_cluster(box(0.3, 0.3, 0.9, n=10), params)

    def test_wall_segment_rejected_by_single_axis_guard(self, params):
        assert not is_human_cluster(box(3.0, 0.1, 2.0), params)

    def test_low_table_too_short(self, params):
        assert not is_human_cluster(box(0.6, 0.6, 0.5), params)

    def test_cabinet_footprint_too_large(self, params):
        assert not is_human_cluster(box(0.9, 0.9, 1.0), params)

    def test_squat_object_fails_aspect(self, params):
        # footprint ~0.92 m, height 0.9 m -> aspect ~0.98 < 1.2
        assert not is_human_cluster(box(0.7, 0.6, 0.9), params)

    def test_point_count_limits(self, params):
        assert not is_human_cluster(box(0.4, 0.3, 1.7, n=2), params)
        assert not is_human_cluster(box(0.4, 0.3, 1.7, n=params['human_max_pts'] + 1), params)


class TestRepeatedHumanTracker:

    @staticmethod
    def tracker(params):
        return RepeatedHumanTracker(params['temporal_voxel_size'],
                                    params['temporal_window'],
                                    params['temporal_static_thresh'])

    def test_stationary_person_is_confirmed_for_removal(self, params):
        t = self.tracker(params)
        person = box(0.4, 0.3, 1.7)
        for _ in range(params['temporal_window']):
            t.update(person)
        assert t.warmed_up
        assert t.remove_mask(person).all()
        far_wall = box(3.0, 0.1, 2.0, origin=(10.0, 10.0, 0.0))
        assert not t.remove_mask(far_wall).any()

    def test_briefly_seen_cluster_is_not_confirmed(self, params):
        t = self.tracker(params)
        person = box(0.4, 0.3, 1.7)
        empty = np.zeros((0, 3), dtype=np.float32)
        for scan in range(params['temporal_window']):
            t.update(person if scan < 3 else empty)
        assert not t.remove_mask(person).any()

    def test_confirmation_clears_after_the_person_leaves(self, params):
        t = self.tracker(params)
        person = box(0.4, 0.3, 1.7)
        empty = np.zeros((0, 3), dtype=np.float32)
        for _ in range(params['temporal_window']):
            t.update(person)
        for _ in range(params['temporal_window']):
            t.update(empty)
        assert not t.remove_mask(person).any()


def test_voxel_downsample_keeps_one_centroid_per_voxel():
    pts = np.array([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03], [0.55, 0.0, 0.0]], dtype=np.float32)
    out = voxel_downsample(pts, 0.1)
    assert len(out) == 2
    assert np.allclose(sorted(out[:, 0]), [0.02, 0.55])


def test_pointcloud2_round_trip():
    xyz = box(1.0, 2.0, 3.0, n=50)
    back = pc2_to_xyz(xyz_to_pc2(xyz, Header()))
    assert np.array_equal(back, xyz)
