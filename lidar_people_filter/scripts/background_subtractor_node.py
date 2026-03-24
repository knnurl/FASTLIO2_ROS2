#!/usr/bin/env python3
"""
Static Map Background Subtractor Node
--------------------------------------
Builds an occupancy voxel map of the static environment and removes
any dynamic foreground points (people, vehicles, etc.) that weren't
present during an initial "empty scene" calibration.

This complements the geometry-based people_filter_node by catching
dynamic objects that don't fit the human silhouette model.

Usage:
  1. Launch with the environment clear of people.
  2. Call the /background_subtractor/calibrate service to capture background.
  3. All subsequent scans are filtered against the background voxel map.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_srvs.srv import Trigger
from std_msgs.msg import Header

import numpy as np
from collections import defaultdict


# ── Re-use PointCloud2 helpers from people_filter_node ──────────────────────

DTYPE_MAP = {
    PointField.FLOAT32: np.float32,
    PointField.FLOAT64: np.float64,
    PointField.INT8:    np.int8,
    PointField.INT16:   np.int16,
    PointField.INT32:   np.int32,
    PointField.UINT8:   np.uint8,
    PointField.UINT16:  np.uint16,
    PointField.UINT32:  np.uint32,
}


def pc2_to_xyz(msg):
    offsets = {f.name: f.offset for f in msg.fields}
    dtypes  = {f.name: DTYPE_MAP.get(f.datatype, np.float32) for f in msg.fields}
    n_pts   = msg.width * msg.height
    ps      = msg.point_step
    raw     = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    raw     = raw[:n_pts * ps].reshape(n_pts, ps)

    def read_field(name):
        off  = offsets[name]
        dt   = dtypes[name]
        size = np.dtype(dt).itemsize
        return np.frombuffer(raw[:, off:off + size].tobytes(), dtype=dt)

    xyz   = np.stack([read_field('x'), read_field('y'), read_field('z')], axis=1).astype(np.float32)
    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid]


def xyz_to_pc2(header, points):
    msg            = PointCloud2()
    msg.header     = header
    msg.height     = 1
    msg.width      = len(points)
    msg.is_dense   = True
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step   = 12 * len(points)
    msg.fields     = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = points.astype(np.float32).tobytes()
    return msg


# ────────────────────────────────────────────────────────────────────────────

class VoxelMap:
    """
    3-D voxel occupancy map.
    Tracks hit counts per voxel for probabilistic background modelling.
    """

    def __init__(self, voxel_size: float):
        self.voxel_size   = voxel_size
        self._counts: dict = defaultdict(int)   # voxel_key → hit count

    def _key(self, point: np.ndarray):
        ix, iy, iz = (point / self.voxel_size).astype(np.int32)
        return (int(ix), int(iy), int(iz))

    def insert(self, points: np.ndarray):
        keys = (points / self.voxel_size).astype(np.int32)
        for k in keys:
            self._counts[(k[0], k[1], k[2])] += 1

    def is_background(self, point: np.ndarray, min_hits: int = 1) -> bool:
        return self._counts.get(self._key(point), 0) >= min_hits

    def filter_foreground(self, points: np.ndarray, min_hits: int = 1) -> np.ndarray:
        """Return only points NOT in the background map."""
        if len(self._counts) == 0:
            return points  # no background yet → pass through
        keys = (points / self.voxel_size).astype(np.int32)
        mask = np.array([
            self._counts.get((k[0], k[1], k[2]), 0) < min_hits
            for k in keys
        ], dtype=bool)
        return points[mask]

    def clear(self):
        self._counts.clear()

    @property
    def num_voxels(self):
        return len(self._counts)


class BackgroundSubtractorNode(Node):

    def __init__(self):
        super().__init__('background_subtractor_node')

        self.declare_parameters('', [
            ('input_topic',  '/points_no_people'),
            ('output_topic', '/points_static'),
            ('voxel_size', 0.1),
            ('calibration_frames', 50),     # frames to accumulate background
            ('min_bg_hits', 3),             # voxel must be seen ≥ N times
            ('auto_calibrate', False),      # start calibrating immediately
        ])

        p = self.get_param

        self._vmap          = VoxelMap(p('voxel_size'))
        self._calibrating   = p('auto_calibrate')
        self._bg_frames     = 0
        self._bg_target     = p('calibration_frames')
        self._bg_ready      = False

        self._pub = self.create_publisher(PointCloud2, p('output_topic'), 10)
        self._sub = self.create_subscription(
            PointCloud2, p('input_topic'), self._callback, 10)

        self._srv_calibrate = self.create_service(
            Trigger, '~/calibrate', self._srv_calibrate_cb)
        self._srv_reset = self.create_service(
            Trigger, '~/reset', self._srv_reset_cb)

        mode = "AUTO-CALIBRATING" if self._calibrating else "PASSTHROUGH (call ~/calibrate to begin)"
        self.get_logger().info(
            f"BackgroundSubtractorNode ready | mode: {mode}\n"
            f"  input  → {p('input_topic')}\n"
            f"  output → {p('output_topic')}"
        )

    def get_param(self, name):
        return self.get_parameter(name).value

    # ── Calibration services ───────────────────────────────────────────────

    def _srv_calibrate_cb(self, _req, response):
        self._vmap.clear()
        self._bg_frames   = 0
        self._calibrating = True
        self._bg_ready    = False
        response.success = True
        response.message = f"Calibration started ({self._bg_target} frames)"
        self.get_logger().info(response.message)
        return response

    def _srv_reset_cb(self, _req, response):
        self._vmap.clear()
        self._calibrating = False
        self._bg_ready    = False
        self._bg_frames   = 0
        response.success = True
        response.message = "Background map cleared"
        self.get_logger().warn(response.message)
        return response

    # ── Main callback ──────────────────────────────────────────────────────

    def _callback(self, msg: PointCloud2):
        try:
            xyz = pc2_to_xyz(msg)
        except Exception as exc:
            self.get_logger().error(str(exc))
            return

        if self._calibrating:
            self._vmap.insert(xyz)
            self._bg_frames += 1
            if self._bg_frames >= self._bg_target:
                self._calibrating = False
                self._bg_ready    = True
                self.get_logger().info(
                    f"Background calibration complete: "
                    f"{self._vmap.num_voxels} voxels from {self._bg_frames} frames"
                )
            # During calibration, publish the raw cloud unchanged
            self._pub.publish(xyz_to_pc2(msg.header, xyz))
            return

        if not self._bg_ready:
            # No background → pass through
            self._pub.publish(xyz_to_pc2(msg.header, xyz))
            return

        foreground = self._vmap.filter_foreground(xyz, min_hits=self.get_param('min_bg_hits'))
        self._pub.publish(xyz_to_pc2(msg.header, foreground))
        self.get_logger().debug(
            f"BG subtraction: {len(xyz)} → {len(foreground)} pts "
            f"({len(xyz)-len(foreground)} removed)"
        )


def main(args=None):
    rclpy.init(args=args)
    node = BackgroundSubtractorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
