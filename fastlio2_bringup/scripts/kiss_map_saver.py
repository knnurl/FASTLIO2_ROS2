#!/usr/bin/env python3
"""
kiss_map_saver.py — Map saver for KISS-ICP on Spot
Subscribes to /kiss/local_map and saves the latest cloud to PCD on service call.
Exposes /kiss/save_map (interface/srv/SaveMaps) — same interface as fastlio2.

Prerequisites:
    source ~/ros2_ws/install/setup.bash
    source ~/fastlio2_ws/install/setup.bash

Run standalone:
    ros2 run fastlio2_bringup kiss_map_saver.py

The node will print a warning until the first /kiss/local_map message arrives.
publish_debug_clouds must be true in the KISS-ICP node for this topic to exist.
"""
import os
import struct
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2

from interface.srv import SaveMaps


class KissMapSaver(Node):
    def __init__(self):
        super().__init__('kiss_map_saver')

        self._lock       = threading.Lock()
        self._cloud_xyz  = None   # (N,3) float32 numpy array, latest map cloud
        self._map_topic  = '/kiss/local_map'
        self._svc_topic  = '/kiss/save_map'

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self._sub = self.create_subscription(
            PointCloud2, self._map_topic, self._cloud_cb, qos)
        self._srv = self.create_service(
            SaveMaps, self._svc_topic, self._save_cb)

        self.get_logger().info(
            f'kiss_map_saver ready.\n'
            f'  Listening : {self._map_topic}\n'
            f'  Service   : {self._svc_topic} (interface/srv/SaveMaps)\n'
            f'Make sure kiss_icp_node has publish_debug_clouds: true'
        )

    # ── PointCloud2 subscriber ────────────────────────────────────────────────

    def _cloud_cb(self, msg: PointCloud2):
        """Parse incoming PointCloud2 into a (N,3) float32 numpy array."""
        try:
            xyz = self._pc2_to_xyz(msg)
            with self._lock:
                self._cloud_xyz = xyz
        except Exception as exc:
            self.get_logger().warn(f'Failed to parse cloud: {exc}')

    @staticmethod
    def _pc2_to_xyz(msg: PointCloud2) -> np.ndarray:
        """Extract x/y/z from a PointCloud2 message by field offset."""
        x_off = y_off = z_off = -1
        for f in msg.fields:
            if f.name == 'x':   x_off = f.offset
            elif f.name == 'y': y_off = f.offset
            elif f.name == 'z': z_off = f.offset
        if x_off < 0 or y_off < 0 or z_off < 0:
            raise ValueError('PointCloud2 missing x/y/z fields')

        step    = msg.point_step
        n_pts   = msg.width * msg.height
        data    = bytes(msg.data)
        fmt     = f'{n_pts}f'
        x_vals  = np.frombuffer(data, dtype=np.float32,
                                count=n_pts, offset=x_off).copy()
        y_vals  = np.frombuffer(data, dtype=np.float32,
                                count=n_pts, offset=y_off).copy()
        z_vals  = np.frombuffer(data, dtype=np.float32,
                                count=n_pts, offset=z_off).copy()

        # frombuffer with step != 4 requires manual stride extraction
        if step != 4:
            x_vals = np.array([
                struct.unpack_from('f', data, i * step + x_off)[0]
                for i in range(n_pts)], dtype=np.float32)
            y_vals = np.array([
                struct.unpack_from('f', data, i * step + y_off)[0]
                for i in range(n_pts)], dtype=np.float32)
            z_vals = np.array([
                struct.unpack_from('f', data, i * step + z_off)[0]
                for i in range(n_pts)], dtype=np.float32)

        xyz = np.stack([x_vals, y_vals, z_vals], axis=1)
        mask = np.isfinite(xyz).all(axis=1)
        return xyz[mask]

    # ── SaveMaps service ──────────────────────────────────────────────────────

    def _save_cb(self, request: SaveMaps.Request,
                 response: SaveMaps.Response) -> SaveMaps.Response:
        with self._lock:
            xyz = self._cloud_xyz

        if xyz is None:
            response.success = False
            response.message = (
                f'No map data received yet on {self._map_topic}. '
                f'Is kiss_icp_node running with publish_debug_clouds: true?'
            )
            self.get_logger().error(response.message)
            return response

        if len(xyz) == 0:
            response.success = False
            response.message = 'Map cloud is empty.'
            self.get_logger().error(response.message)
            return response

        path = request.file_path.strip() or \
               os.path.expanduser('~/fastlio2_ws/maps/map_spot_neric.pcd')

        # Rename existing file with timestamp (matches fastlio2 behaviour)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            from datetime import datetime
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stamped = p.parent / f'{p.stem}_{stamp}.pcd'
            path = str(stamped)
            self.get_logger().info(
                f'File exists — saving as {path}')

        try:
            self._write_pcd(xyz, path)
            response.success = True
            response.message = f'Saved {len(xyz)} points to {path}'
            self.get_logger().info(response.message)
        except Exception as exc:
            response.success = False
            response.message = f'Failed to write PCD: {exc}'
            self.get_logger().error(response.message)

        return response

    @staticmethod
    def _write_pcd(xyz: np.ndarray, path: str):
        """Write a (N,3) float32 array to a binary PCD file (PCL-compatible)."""
        n = len(xyz)
        header = (
            f'# .PCD v0.7 — kiss_map_saver\n'
            f'VERSION 0.7\n'
            f'FIELDS x y z\n'
            f'SIZE 4 4 4\n'
            f'TYPE F F F\n'
            f'COUNT 1 1 1\n'
            f'WIDTH {n}\n'
            f'HEIGHT 1\n'
            f'VIEWPOINT 0 0 0 1 0 0 0\n'
            f'POINTS {n}\n'
            f'DATA binary\n'
        )
        with open(path, 'wb') as f:
            f.write(header.encode('ascii'))
            f.write(xyz.astype(np.float32).tobytes())


def main():
    rclpy.init()
    node = KissMapSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
