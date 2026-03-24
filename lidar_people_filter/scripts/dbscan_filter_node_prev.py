#!/usr/bin/env python3
"""
dbscan_filter_node.py — World-frame people filter with frame accumulation.

Subscribes to FASTLIO2's /fastlio2/world_cloud (PointCloud2, world/map frame).
Accumulates N frames to compensate for the MID360's non-repetitive scan
pattern — each single frame is sparse; N frames give much denser clusters.
Runs DBSCAN on the XY plane, classifies clusters by height + footprint diagonal
+ aspect ratio (height / footprint), removes human-shaped ones.
Publishes filtered cloud to /fastlio2/world_cloud_filtered.

Why world frame instead of body frame:
  - Accumulated frames all share the same coordinate system (Z = gravity up),
    so DBSCAN height checks are geometrically correct without any transforms.
  - No RANSAC ground removal needed — a simple Z-range strip is sufficient.
  - The localizer ICP path (body_cloud → people_filter_node) is separate and
    unaffected by this node.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

import numpy as np
from collections import deque

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    raise ImportError("scikit-learn required: pip3 install scikit-learn")


# ── PointCloud2 helpers ───────────────────────────────────────────────────────

def pc2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Extract (N, 3) float32 XYZ from a PointCloud2 message."""
    field_map = {f.name: f.offset for f in msg.fields}
    ox, oy, oz = field_map.get('x', 0), field_map.get('y', 4), field_map.get('z', 8)
    step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8)
    n = msg.width * msg.height
    pts = np.empty((n, 3), dtype=np.float32)
    # Vectorised extraction using stride tricks
    base = np.arange(n) * step
    pts[:, 0] = np.frombuffer(
        np.stack([data[base + ox + k] for k in range(4)], axis=1).tobytes(),
        dtype=np.float32)
    pts[:, 1] = np.frombuffer(
        np.stack([data[base + oy + k] for k in range(4)], axis=1).tobytes(),
        dtype=np.float32)
    pts[:, 2] = np.frombuffer(
        np.stack([data[base + oz + k] for k in range(4)], axis=1).tobytes(),
        dtype=np.float32)
    return pts[~np.isnan(pts).any(axis=1)]


def xyz_to_pc2(xyz: np.ndarray, header: Header) -> PointCloud2:
    """Pack (N, 3) float32 array into a PointCloud2 message."""
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(xyz)
    msg.is_bigendian = False
    msg.point_step = 12   # 3 × float32
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = xyz.astype(np.float32).tobytes()
    return msg


# ── Human classifier ──────────────────────────────────────────────────────────

def is_human(pts: np.ndarray, p: dict) -> bool:
    """
    Return True if a cluster matches human geometry.

    Checks (in short-circuit order, cheapest first):
      1. Point count in [min_pts, max_pts]
      2. Vertical extent in [min_height, max_height]
      3. Footprint diagonal in [min_footprint, max_footprint]
      4. Aspect ratio = height / footprint ≥ min_aspect
         (people are tall and narrow relative to their footprint)
    """
    n = len(pts)
    if n < p['min_pts'] or n > p['max_pts']:
        return False

    z_min, z_max = float(pts[:, 2].min()), float(pts[:, 2].max())
    height = z_max - z_min
    if height < p['min_height'] or height > p['max_height']:
        return False

    x_range = float(pts[:, 0].max() - pts[:, 0].min())
    y_range = float(pts[:, 1].max() - pts[:, 1].min())
    footprint = np.sqrt(x_range ** 2 + y_range ** 2)
    if footprint < p['min_footprint'] or footprint > p['max_footprint']:
        return False

    if footprint > 0 and (height / footprint) < p['min_aspect']:
        return False

    return True


# ── Node ─────────────────────────────────────────────────────────────────────

class DBSCANFilterNode(Node):

    def __init__(self):
        super().__init__('dbscan_filter_node')

        # ── parameters ──────────────────────────────────────────────────────
        self.declare_parameter('input_topic',  '/fastlio2/world_cloud')
        self.declare_parameter('output_topic', '/fastlio2/world_cloud_filtered')

        # Frame accumulation — MID360 non-repetitive pattern: single ~100 ms
        # frame is sparse. Accumulating 4 frames (~400 ms) gives enough density
        # for reliable DBSCAN clusters. Reduce to 2 for faster response; increase
        # to 6 in large/open environments.
        self.declare_parameter('accumulate_frames', 4)

        # DBSCAN neighbourhood radius (metres, XY plane only).
        #   Too small → person splits into fragments, missed.
        #   Too large → nearby wall merges into one person-sized cluster.
        # 0.25 m works well for MID360 at 1–8 m range indoors.
        self.declare_parameter('dbscan_eps',         0.25)
        self.declare_parameter('dbscan_min_samples', 10)

        # World-frame Z strip — removes ground and ceiling. Because this node
        # works in world frame (Z = gravity up), simple Z bounds replace RANSAC.
        #   ground_z_min: below this is ground. Adjust if sensor is mounted high.
        #   ground_z_max: above this is ceiling / tall structure, not a person.
        self.declare_parameter('ground_z_min', -0.3)   # m above map origin
        self.declare_parameter('ground_z_max',  2.5)   # m above map origin

        # Human geometry bounds
        self.declare_parameter('min_height',    1.2)   # m — lower catches leg-clipped people
        self.declare_parameter('max_height',    2.2)   # m
        self.declare_parameter('min_footprint', 0.15)  # m diagonal — avoids thin poles
        self.declare_parameter('max_footprint', 1.0)   # m diagonal — person + heavy clothing
        self.declare_parameter('min_pts',       15)    # after accumulation
        self.declare_parameter('max_pts',       2500)  # wall segments will exceed this
        # height / footprint ratio. Humans ≈ 2.0+. Boxes, chairs < 1.5.
        self.declare_parameter('min_aspect',    1.8)

        # ── state ───────────────────────────────────────────────────────────
        n_frames = self.get_parameter('accumulate_frames').value
        self._buffer: deque = deque(maxlen=n_frames)
        self._latest_header = Header()

        # ── pub / sub ───────────────────────────────────────────────────────
        in_topic  = self.get_parameter('input_topic').value
        out_topic = self.get_parameter('output_topic').value

        self._sub = self.create_subscription(
            PointCloud2, in_topic, self._cloud_cb, 10)
        self._pub = self.create_publisher(
            PointCloud2, out_topic, 10)

        self.get_logger().info(
            f"DBSCAN world filter ready — "
            f"in={in_topic}  out={out_topic}  "
            f"acc={n_frames} frames")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _params(self) -> dict:
        g = self.get_parameter
        return {
            'eps':           g('dbscan_eps').value,
            'min_samples':   g('dbscan_min_samples').value,
            'ground_z_min':  g('ground_z_min').value,
            'ground_z_max':  g('ground_z_max').value,
            'min_height':    g('min_height').value,
            'max_height':    g('max_height').value,
            'min_footprint': g('min_footprint').value,
            'max_footprint': g('max_footprint').value,
            'min_pts':       g('min_pts').value,
            'max_pts':       g('max_pts').value,
            'min_aspect':    g('min_aspect').value,
        }

    # ── callback ──────────────────────────────────────────────────────────────

    def _cloud_cb(self, msg: PointCloud2):
        p = self._params()
        self._latest_header = msg.header

        # 1. Unpack and Z-strip (ground + ceiling removal; works correctly in
        #    world frame because Z is always gravity-up).
        pts = pc2_to_xyz(msg)
        if len(pts) == 0:
            return
        mask = (pts[:, 2] > p['ground_z_min']) & (pts[:, 2] < p['ground_z_max'])
        pts = pts[mask]
        if len(pts) == 0:
            return

        # 2. Accumulate frames — density compensation for MID360's non-repetitive
        #    pattern. All frames share the world frame coordinate system, so the
        #    accumulated cloud is geometrically consistent for clustering.
        self._buffer.append(pts)
        if len(self._buffer) < self._buffer.maxlen:
            return   # wait for buffer to fill (warm-up period)

        acc = np.vstack(self._buffer)

        # 3. DBSCAN on XY plane only. Height is handled by the geometry checks
        #    in is_human(), so we cluster horizontally and test vertically.
        labels = DBSCAN(
            eps=p['eps'],
            min_samples=p['min_samples'],
            n_jobs=-1,
            algorithm='ball_tree',
        ).fit_predict(acc[:, :2])

        # 4. Classify clusters — mask out human-shaped ones.
        human_mask = np.zeros(len(acc), dtype=bool)
        n_humans = 0
        for lbl in set(labels):
            if lbl < 0:
                continue   # DBSCAN noise — always keep
            idx = labels == lbl
            if is_human(acc[idx], p):
                human_mask[idx] = True
                n_humans += 1

        # 5. Publish filtered cloud.
        clean = acc[~human_mask]
        out = xyz_to_pc2(clean, self._latest_header)
        self._pub.publish(out)

        if n_humans > 0:
            self.get_logger().debug(
                f"Removed {n_humans} human cluster(s) "
                f"({human_mask.sum()} pts) — {len(clean)} pts remaining")


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = DBSCANFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
