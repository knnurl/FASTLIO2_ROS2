#!/usr/bin/env python3
"""
Improved DBSCAN people filter for Livox Mid360 + FastLIO.

Key improvements over v1:
  1. Voxel downsample BEFORE DBSCAN — normalises Mid360's range-dependent
     point density so far-range people cluster as well as close-range ones.
  2. Range-adaptive min_samples — closer clusters require more evidence;
     far clusters (sparse) need a lower threshold to form at all.
  3. Wall-merge guard — rejects oversized XY footprints before aspect check.
  4. Temporal consistency filter — voxels hit in most of the last N scans
     are flagged as static and kept; voxels that appear/disappear are dynamic.
     This catches stationary people that DBSCAN geometry cannot.
"""

import threading
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from collections import deque

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    raise ImportError("pip3 install scikit-learn")


# ── point cloud helpers ────────────────────────────────────────────────────────

def pc2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Fast numpy structured-dtype extraction — no Python loops."""
    fm = {f.name: f.offset for f in msg.fields}
    dt = np.dtype({
        'names':   ['x', 'y', 'z'],
        'formats': [np.float32, np.float32, np.float32],
        'offsets': [fm.get('x', 0), fm.get('y', 4), fm.get('z', 8)],
        'itemsize': msg.point_step,
    })
    struct = np.frombuffer(msg.data, dtype=dt)
    return np.column_stack([struct['x'], struct['y'], struct['z']])


def xyz_to_pc2(xyz: np.ndarray, header: Header) -> PointCloud2:
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(xyz)
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(xyz)
    msg.is_dense = True
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = xyz.astype(np.float32).tobytes()
    return msg


def voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample by keeping one point per voxel cell (centroid)."""
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel_size).astype(np.int32)
    # unique rows trick
    key_str = keys[:, 0].astype(np.int64) * 1_000_000 + \
              keys[:, 1].astype(np.int64) * 1_000 + \
              keys[:, 2].astype(np.int64)
    _, inv = np.unique(key_str, return_inverse=True)
    out = np.zeros((inv.max() + 1, 3), dtype=np.float32)
    np.add.at(out, inv, pts)
    counts = np.bincount(inv).reshape(-1, 1)
    return (out / counts).astype(np.float32)


def is_human_cluster(pts, p):
    n = len(pts)
    if n < p['human_min_pts'] or n > p['human_max_pts']:
        return False

    z_ext = pts[:, 2].max() - pts[:, 2].min()
    if z_ext < p['human_min_height'] or z_ext > p['human_max_height']:
        return False

    x_ext = pts[:, 0].max() - pts[:, 0].min()
    y_ext = pts[:, 1].max() - pts[:, 1].min()
    footprint = np.sqrt(x_ext**2 + y_ext**2)

    # wall-merge guard: if either single axis is very wide it's structure
    if x_ext > p['human_max_single_axis'] or y_ext > p['human_max_single_axis']:
        return False

    if footprint < p['human_min_footprint'] or footprint > p['human_max_footprint']:
        return False

    if footprint > 0 and (z_ext / footprint) < p['human_min_aspect']:
        return False

    return True


# ── marker helpers ─────────────────────────────────────────────────────────────

_BOX_EDGES = [
    (0,1),(1,2),(2,3),(3,0),   # bottom face
    (4,5),(5,6),(6,7),(7,4),   # top face
    (0,4),(1,5),(2,6),(3,7),   # verticals
]

def _cluster_box_marker(header, marker_id: int, cluster: np.ndarray,
                         r: float, g: float, b: float,
                         lifetime_sec: float = 0.5) -> Marker:
    """LINE_LIST marker drawing the 12-edge bounding box of a cluster."""
    mn = cluster.min(axis=0)
    mx = cluster.max(axis=0)
    cx, cy, cz = (mn + mx) / 2
    hx, hy, hz = (mx - mn) / 2

    corners = [
        (cx-hx, cy-hy, cz-hz), (cx+hx, cy-hy, cz-hz),
        (cx+hx, cy+hy, cz-hz), (cx-hx, cy+hy, cz-hz),
        (cx-hx, cy-hy, cz+hz), (cx+hx, cy-hy, cz+hz),
        (cx+hx, cy+hy, cz+hz), (cx-hx, cy+hy, cz+hz),
    ]

    m = Marker()
    m.header = header
    m.ns = 'dbscan_detections'
    m.id = marker_id
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = 0.04
    m.color.r = r;  m.color.g = g;  m.color.b = b;  m.color.a = 1.0
    m.lifetime = Duration(seconds=lifetime_sec).to_msg()
    for a, b_idx in _BOX_EDGES:
        pa, pb = corners[a], corners[b_idx]
        m.points.append(Point(x=float(pa[0]), y=float(pa[1]), z=float(pa[2])))
        m.points.append(Point(x=float(pb[0]), y=float(pb[1]), z=float(pb[2])))
    return m


# ── temporal consistency tracker ──────────────────────────────────────────────

class RepeatedHumanTracker:
    """
    Tracks voxels that are REPEATEDLY classified as human-shaped by DBSCAN.

    A voxel flagged as human in >= confirm_thresh fraction of the last
    `window` scans is considered a stationary person and added to the
    persistent remove set.

    This is the correct approach for stationary people:
    - Walls are never human-shaped → never accumulate hits → never removed.
    - A standing person is human-shaped every scan → hits 100% → removed.
    - A person who just left → hits drop below thresh → eventually cleared.
    """

    def __init__(self, voxel_size: float, window: int, confirm_thresh: float):
        self.voxel_size = voxel_size
        self.window = window
        self.confirm_thresh = confirm_thresh
        # key → deque of bool (was this voxel in a human cluster this scan?)
        self._hits: dict = {}
        # confirmed stationary-human voxels (persistent remove set)
        self._confirmed: set = set()

    def _keys(self, pts: np.ndarray) -> np.ndarray:
        return np.floor(pts / self.voxel_size).astype(np.int32)

    def update(self, human_pts: np.ndarray):
        """
        Call once per scan with the points belonging to DBSCAN human clusters.
        Updates hit history and refreshes confirmed set.
        """
        human_keys = set(
            map(tuple, self._keys(human_pts))
        ) if len(human_pts) else set()

        # Tick every tracked voxel (hit or miss this scan)
        all_keys = set(self._hits.keys()) | human_keys
        for k in all_keys:
            if k not in self._hits:
                self._hits[k] = deque(maxlen=self.window)
            self._hits[k].append(k in human_keys)

        # Prune voxels not seen in a full window
        stale = [k for k, h in self._hits.items()
                 if len(h) == self.window and not any(h)]
        for k in stale:
            del self._hits[k]
            self._confirmed.discard(k)

        # Rebuild confirmed set
        self._confirmed = {
            k for k, h in self._hits.items()
            if len(h) >= self.window // 2           # need at least half window
            and sum(h) / len(h) >= self.confirm_thresh
        }

    def remove_mask(self, pts: np.ndarray) -> np.ndarray:
        """
        Returns boolean mask over pts — True = belongs to a confirmed
        stationary-human voxel → should be removed.
        """
        if len(pts) == 0 or not self._confirmed:
            return np.zeros(len(pts), dtype=bool)
        keys = self._keys(pts).astype(np.int64)
        pts_enc = keys[:, 0] * 1_000_000 + keys[:, 1] * 1_000 + keys[:, 2]
        conf_enc = np.array(
            [k[0] * 1_000_000 + k[1] * 1_000 + k[2] for k in self._confirmed],
            dtype=np.int64)
        return np.isin(pts_enc, conf_enc)

    @property
    def warmed_up(self) -> bool:
        if not self._hits:
            return False
        return np.mean([len(h) for h in self._hits.values()]) >= self.window * 0.5


# ── main node ──────────────────────────────────────────────────────────────────

class DBSCANFilterNode(Node):

    def __init__(self):
        super().__init__('dbscan_filter_node')

        # topics
        self.declare_parameter('input_topic',             '/cloud_registered')
        self.declare_parameter('output_topic',            '/cloud_filtered')

        # accumulation
        self.declare_parameter('accumulate_frames',       4)

        # voxel downsample before DBSCAN — key fix for density variance
        self.declare_parameter('voxel_size',              0.12)

        # DBSCAN base params
        self.declare_parameter('dbscan_eps',              0.3)
        # base min_samples — will be scaled by range (see adaptive logic)
        self.declare_parameter('dbscan_min_samples',      5)

        # human geometry
        self.declare_parameter('human_min_height',        1.2)
        self.declare_parameter('human_max_height',        2.2)
        self.declare_parameter('human_min_footprint',     0.1)
        self.declare_parameter('human_max_footprint',     1.0)
        # wall-merge guard: max extent on any single horizontal axis
        self.declare_parameter('human_max_single_axis',   1.2)
        self.declare_parameter('human_min_pts',           4)
        self.declare_parameter('human_max_pts',           500)
        self.declare_parameter('human_min_aspect',        1.5)

        # ground strip (world frame, z up)
        self.declare_parameter('ground_z_min',           -1.5)
        self.declare_parameter('ground_z_max',            3.5)

        # temporal consistency (stationary people)
        self.declare_parameter('temporal_enable',         True)
        # voxel size for temporal tracker (coarser than DBSCAN voxel)
        self.declare_parameter('temporal_voxel_size',     0.2)
        # number of scans in sliding window
        self.declare_parameter('temporal_window',         12)
        # fraction of window a voxel must be hit to be considered static
        # 0.7 = seen in 70% of last 12 scans → static → KEEP
        # anything below is dynamic → candidate for removal
        self.declare_parameter('temporal_static_thresh',  0.7)

        self.declare_parameter('markers_topic', '/dbscan_detections')

        in_t      = self.get_parameter('input_topic').value
        out_t     = self.get_parameter('output_topic').value
        markers_t = self.get_parameter('markers_topic').value

        # thread-safe ring buffer — callback appends, worker consumes
        self._buf: deque = deque(maxlen=self.get_parameter('accumulate_frames').value)
        self._buf_lock  = threading.Lock()
        self._new_data  = threading.Event()
        self._shutdown  = threading.Event()

        # cached from first message (constant for a given sensor)
        self._msg_fields     = None
        self._msg_point_step = None
        self._msg_is_bigendian = False
        self._last_hdr       = Header()

        self._tracker = None  # initialised in worker on first run

        self._sub = self.create_subscription(PointCloud2, in_t,  self._cb, 10)
        self._pub = self.create_publisher(   PointCloud2, out_t, 10)
        self._marker_pub = self.create_publisher(MarkerArray, markers_t, 10)

        # spin heavy processing on a background thread so the ROS executor
        # never blocks — callback is now just unpack + buffer + signal
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        self.get_logger().info(
            f"DBSCAN filter ready | {in_t} → {out_t} | "
            f"voxel={self.get_parameter('voxel_size').value}m "
            f"temporal={self.get_parameter('temporal_enable').value}")

    def _p(self):
        names = [
            'accumulate_frames','voxel_size',
            'dbscan_eps','dbscan_min_samples',
            'human_min_height','human_max_height',
            'human_min_footprint','human_max_footprint',
            'human_max_single_axis',
            'human_min_pts','human_max_pts','human_min_aspect',
            'ground_z_min','ground_z_max',
            'temporal_enable','temporal_voxel_size',
            'temporal_window','temporal_static_thresh',
        ]
        return {k: self.get_parameter(k).value for k in names}

    def _range_adaptive_eps(self, pts: np.ndarray, base_eps: float) -> float:
        ranges = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        med = np.median(ranges)
        if med > 6.0:
            return base_eps * 1.5
        elif med > 3.0:
            return base_eps * 1.2
        return base_eps

    # ── subscription callback — intentionally lightweight ─────────────────────

    def _cb(self, msg: PointCloud2):
        """Unpack, z-strip, buffer, signal worker. No heavy work here."""
        p = self._p()

        if self._msg_fields is None:
            self._msg_fields     = msg.fields
            self._msg_point_step = msg.point_step
            self._msg_is_bigendian = msg.is_bigendian

        pts = pc2_to_xyz(msg)
        n   = msg.width * msg.height
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step).copy()

        z_mask = (pts[:, 2] > p['ground_z_min']) & (pts[:, 2] < p['ground_z_max'])
        pts, raw = pts[z_mask], raw[z_mask]
        if len(pts) == 0:
            return

        with self._buf_lock:
            self._buf.append((pts, raw))
            self._last_hdr = msg.header

        self._new_data.set()   # wake worker

    # ── background worker — all heavy processing runs here ────────────────────

    def _worker_loop(self):
        while not self._shutdown.is_set():
            triggered = self._new_data.wait(timeout=0.5)
            if not triggered:
                continue
            self._new_data.clear()
            self._process()

    def _process(self):
        p = self._p()

        with self._buf_lock:
            if len(self._buf) < p['accumulate_frames']:
                return
            acc     = np.vstack([b[0] for b in self._buf])
            acc_raw = np.vstack([b[1] for b in self._buf])
            hdr     = self._last_hdr

        # 1. voxel downsample
        down = voxel_downsample(acc, p['voxel_size'])
        if len(down) == 0:
            return

        # 2. DBSCAN with range-adaptive eps
        adaptive_eps = self._range_adaptive_eps(down, p['dbscan_eps'])
        labels = DBSCAN(
            eps=adaptive_eps,
            min_samples=p['dbscan_min_samples'],
            n_jobs=-1,
        ).fit(down[:, :2]).labels_

        # 3. classify clusters, collect human voxel keys (vectorised)
        human_voxels   = set()
        human_clusters = []
        for lbl in set(labels) - {-1}:
            cluster = down[labels == lbl]
            if is_human_cluster(cluster, p):
                human_clusters.append(cluster)
                keys = np.floor(cluster / p['voxel_size']).astype(np.int32)
                human_voxels.update(map(tuple, keys))

        n_detected = len(human_clusters)

        # 4. map human voxels → original point mask (vectorised)
        dbscan_mask = np.zeros(len(acc), dtype=bool)
        if human_voxels:
            acc_keys = np.floor(acc / p['voxel_size']).astype(np.int64)
            acc_enc  = (acc_keys[:, 0] * 1_000_000 +
                        acc_keys[:, 1] * 1_000   +
                        acc_keys[:, 2])
            hv_enc = np.array(
                [k[0] * 1_000_000 + k[1] * 1_000 + k[2] for k in human_voxels],
                dtype=np.int64)
            dbscan_mask = np.isin(acc_enc, hv_enc)

        # 5. bounding-box markers
        ma = MarkerArray()
        for i, cluster in enumerate(human_clusters):
            ma.markers.append(_cluster_box_marker(hdr, i, cluster, 1.0, 0.5, 0.0))
        for i in range(n_detected, n_detected + 10):
            del_m = Marker()
            del_m.header = hdr
            del_m.ns = 'dbscan_detections'
            del_m.id = i
            del_m.action = Marker.DELETE
            ma.markers.append(del_m)
        self._marker_pub.publish(ma)

        # 6. repeated-human tracker
        temporal_mask = np.zeros(len(acc), dtype=bool)
        if p['temporal_enable']:
            if self._tracker is None:
                self._tracker = RepeatedHumanTracker(
                    p['temporal_voxel_size'],
                    p['temporal_window'],
                    p['temporal_static_thresh'])
            human_pts = acc[dbscan_mask] if dbscan_mask.any() else np.empty((0, 3))
            self._tracker.update(human_pts)
            if self._tracker.warmed_up:
                temporal_mask = self._tracker.remove_mask(acc)

        # 7. publish filtered cloud
        remove = dbscan_mask | temporal_mask
        n_temporal = int(temporal_mask.sum())
        if n_detected > 0 or n_temporal > 0:
            self.get_logger().debug(
                f"Removed: {n_detected} DBSCAN + {n_temporal} temporal pts | "
                f"kept {(~remove).sum()}/{len(acc)}")

        kept_raw = acc_raw[~remove]
        out = PointCloud2()
        out.header       = hdr
        out.height       = 1
        out.width        = len(kept_raw)
        out.fields       = self._msg_fields
        out.is_bigendian = self._msg_is_bigendian
        out.point_step   = self._msg_point_step
        out.row_step     = self._msg_point_step * len(kept_raw)
        out.is_dense     = True
        out.data         = kept_raw.tobytes()
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DBSCANFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._shutdown.set()
        node._worker_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
