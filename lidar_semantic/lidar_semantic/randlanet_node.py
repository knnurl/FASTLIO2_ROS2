#!/usr/bin/env python3
"""
randlanet_node.py — ROS2 node for RandLA-Net semantic segmentation.

Subscribes to a PointCloud2, runs RandLA-Net inference in a background
thread, and publishes:
  • /semantic/labelled_cloud   — full cloud, intensity = class index
  • /semantic/person_cloud     — person points only
  • /semantic/ground_cloud     — road/parking/sidewalk/terrain points
  • /semantic/markers          — class-coloured sphere markers

Designed to run alongside the FastLIO2 + DBSCAN people filter stack.
"""

import threading
import struct
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

from .model.randlanet import RandLANet, knn_query, random_downsample


# ── SemanticKITTI class definitions ──────────────────────────────────────────

LABELS = {
    0:  ('car',           (100, 150, 245)),
    1:  ('bicycle',       (100, 230, 245)),
    2:  ('motorcycle',    (30,  60,  150)),
    3:  ('truck',         (80,  30,  180)),
    4:  ('other-vehicle', (100, 80,  250)),
    5:  ('person',        (255, 30,  30)),    # red — primary detection target
    6:  ('bicyclist',     (255, 40,  200)),
    7:  ('motorcyclist',  (150, 30,  90)),
    8:  ('road',          (255, 0,   255)),
    9:  ('parking',       (255, 150, 255)),
    10: ('sidewalk',      (75,  0,   75)),
    11: ('other-ground',  (75,  0,   175)),
    12: ('building',      (0,   200, 255)),
    13: ('fence',         (50,  120, 255)),
    14: ('vegetation',    (0,   175, 0)),
    15: ('trunk',         (0,   60,  135)),
    16: ('terrain',       (80,  240, 150)),
    17: ('pole',          (150, 240, 255)),
    18: ('traffic-sign',  (0,   0,   255)),
}

# Class groups for filtered output topics
PERSON_CLASSES  = {5, 6, 7}         # person, bicyclist, motorcyclist
GROUND_CLASSES  = {8, 9, 10, 11, 16} # road, parking, sidewalk, other-ground, terrain

# Per-class RGB colour table: index → (r, g, b) in 0–255
COLOUR_TABLE = np.zeros((19, 3), dtype=np.uint8)
for cls, (_, rgb) in LABELS.items():
    COLOUR_TABLE[cls] = rgb


# ── Point cloud helpers ───────────────────────────────────────────────────────

def pc2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Extract xyz as (N, 3) float32 from a PointCloud2 message."""
    fm = {f.name: f.offset for f in msg.fields}
    dt = np.dtype({
        'names':   ['x', 'y', 'z'],
        'formats': [np.float32, np.float32, np.float32],
        'offsets': [fm['x'], fm['y'], fm['z']],
        'itemsize': msg.point_step,
    })
    raw = np.frombuffer(msg.data, dtype=dt)
    pts = np.column_stack([raw['x'], raw['y'], raw['z']])
    return pts[np.isfinite(pts).all(axis=1)]


def make_labelled_cloud(header, points: np.ndarray, labels: np.ndarray) -> PointCloud2:
    """
    Build a PointCloud2 where the 'intensity' field carries the class index.
    Colour is encoded in RGB (packed as float).
    """
    N = len(points)
    colours = COLOUR_TABLE[labels]                  # (N, 3) uint8

    # Pack RGB into a uint32 then reinterpret as float32 (RViz RGB32f trick)
    rgb_u32 = (colours[:, 0].astype(np.uint32) << 16 |
               colours[:, 1].astype(np.uint32) << 8  |
               colours[:, 2].astype(np.uint32))
    rgb_f32 = rgb_u32.view(np.float32)

    data = np.zeros(N, dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('rgb', np.float32), ('label', np.float32),
    ])
    data['x']     = points[:, 0]
    data['y']     = points[:, 1]
    data['z']     = points[:, 2]
    data['rgb']   = rgb_f32
    data['label'] = labels.astype(np.float32)

    msg               = PointCloud2()
    msg.header        = header
    msg.height        = 1
    msg.width         = N
    msg.is_dense      = True
    msg.is_bigendian  = False
    msg.point_step    = data.dtype.itemsize
    msg.row_step      = msg.point_step * N
    msg.fields = [
        PointField(name='x',     offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',     offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',     offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb',   offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='label', offset=16, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = data.tobytes()
    return msg


def make_class_cloud(header, points: np.ndarray) -> PointCloud2:
    """Build a plain xyz PointCloud2 for a subset of points."""
    N  = len(points)
    dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
    data         = np.zeros(N, dtype=dt)
    data['x']    = points[:, 0]
    data['y']    = points[:, 1]
    data['z']    = points[:, 2]

    msg              = PointCloud2()
    msg.header       = header
    msg.height       = 1
    msg.width        = N
    msg.is_dense     = True
    msg.is_bigendian = False
    msg.point_step   = 12
    msg.row_step     = 12 * N
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = data.tobytes()
    return msg


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class RandLANetNode(Node):

    def __init__(self):
        super().__init__('randlanet_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('input_topic',    '/fastlio2/body_cloud')
        self.declare_parameter('model_path',     '')
        self.declare_parameter('device',         'cpu')
        self.declare_parameter('num_points',     8192)
        self.declare_parameter('num_neighbors',  16)
        self.declare_parameter('num_classes',    19)
        self.declare_parameter('decimation',            3)   # infer every Nth cloud
        self.declare_parameter('score_threshold',       0.5)
        self.declare_parameter('person_cluster_radius', 0.8)

        self._input_topic   = self.get_parameter('input_topic').value
        self._model_path    = self.get_parameter('model_path').value
        self._device_str    = self.get_parameter('device').value
        self._num_points    = self.get_parameter('num_points').value
        self._num_neighbors = self.get_parameter('num_neighbors').value
        self._num_classes   = self.get_parameter('num_classes').value
        self._decimation      = self.get_parameter('decimation').value
        self._score_thresh    = self.get_parameter('score_threshold').value
        self._cluster_radius  = self.get_parameter('person_cluster_radius').value

        self._device        = torch.device(self._device_str)
        self._frame_count   = 0

        # ── Model ────────────────────────────────────────────────────────────
        self.get_logger().info('Loading RandLA-Net model…')
        self._model = RandLANet(
            d_in=3,
            num_classes=self._num_classes,
            num_neighbors=self._num_neighbors,
        ).to(self._device)

        if self._model_path:
            try:
                self._model = RandLANet.from_pretrained(
                    self._model_path, device=self._device_str,
                    d_in=3, num_classes=self._num_classes,
                    num_neighbors=self._num_neighbors,
                )
                self.get_logger().info(f'Loaded weights from {self._model_path}')
            except Exception as exc:
                self.get_logger().warn(
                    f'Could not load weights from {self._model_path}: {exc}\n'
                    'Running with random weights — output will be meaningless until '
                    'pretrained weights are provided.'
                )
        else:
            self.get_logger().warn(
                'No model_path set — running with random weights.\n'
                'Download pretrained weights and set model_path in semantic.yaml.'
            )

        self._model.eval()

        # ── QoS ─────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5,
        )
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        # ── Pub / Sub ────────────────────────────────────────────────────────
        self._sub = self.create_subscription(
            PointCloud2, self._input_topic, self._cb, sensor_qos)

        self._pub_labelled = self.create_publisher(
            PointCloud2, '/semantic/labelled_cloud', sensor_qos)
        self._pub_person = self.create_publisher(
            PointCloud2, '/semantic/person_cloud', sensor_qos)
        self._pub_ground = self.create_publisher(
            PointCloud2, '/semantic/ground_cloud', sensor_qos)
        self._pub_markers = self.create_publisher(
            MarkerArray, '/semantic/markers', sensor_qos)

        # ── Background thread ────────────────────────────────────────────────
        self._pending_msg  = None
        self._buf_lock     = threading.Lock()
        self._new_data     = threading.Event()
        self._shutdown     = threading.Event()
        self._worker       = threading.Thread(
            target=self._worker_loop, daemon=True, name='randlanet_worker')
        self._worker.start()

        self.get_logger().info(
            f'RandLA-Net node ready  |  '
            f'input={self._input_topic}  device={self._device_str}  '
            f'N={self._num_points}  decimation=1/{self._decimation}'
        )

    # ── Subscription callback (lightweight) ──────────────────────────────────

    def _cb(self, msg: PointCloud2):
        self._frame_count += 1
        if self._frame_count % self._decimation != 0:
            return
        with self._buf_lock:
            self._pending_msg = msg
        self._new_data.set()

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _worker_loop(self):
        while not self._shutdown.is_set():
            if not self._new_data.wait(timeout=0.5):
                continue
            self._new_data.clear()

            with self._buf_lock:
                msg = self._pending_msg

            if msg is None:
                continue
            try:
                self._process(msg)
            except Exception as exc:
                self.get_logger().error(f'Inference error: {exc}', throttle_duration_sec=2.0)

    # ── Inference ─────────────────────────────────────────────────────────────

    def _precompute(self, pts: np.ndarray):
        """
        Build all KNN graphs and downsampling indices for the encoder stages.

        Returns:
            pts_stages:    list of (num_layers+1) [1, N_i, 3] tensors
            knn_list:      list of num_layers      [1, N_i, K] LongTensors
            down_idx_list: list of num_layers      [1, N_{i+1}] LongTensors
        """
        ratios      = [4, 4, 4, 4]
        pts_stages  = [pts]
        knn_list    = []
        down_list   = []

        cur = pts
        for ratio in ratios:
            knn_idx = knn_query(cur, self._num_neighbors)          # (N_i, K)
            knn_list.append(
                torch.from_numpy(knn_idx).long().unsqueeze(0).to(self._device))

            sub_pts, kept = random_downsample(cur, ratio)         # (N_sub,3), (N_sub,)
            down_list.append(
                torch.from_numpy(kept).long().unsqueeze(0).to(self._device))

            pts_stages.append(sub_pts)
            cur = sub_pts

        pts_tensors = [
            torch.from_numpy(p.astype(np.float32)).unsqueeze(0).to(self._device)
            for p in pts_stages
        ]
        return pts_tensors, knn_list, down_list

    def _process(self, msg: PointCloud2):
        # 1. Extract points
        pts = pc2_to_xyz(msg)
        if len(pts) < 64:
            return

        # 2. Subsample / pad to fixed num_points
        if len(pts) > self._num_points:
            idx = np.random.choice(len(pts), self._num_points, replace=False)
            pts = pts[idx]
        elif len(pts) < self._num_points:
            pad = np.random.choice(len(pts), self._num_points - len(pts))
            pts = np.vstack([pts, pts[pad]])

        # 3. Precompute KNN + downsampling for all encoder stages
        pts_tensors, knn_list, down_list = self._precompute(pts)

        # 4. Inference
        with torch.no_grad():
            logits = self._model(pts_tensors, knn_list, down_list)  # (1, C, N)
            probs  = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)                           # (1, N) each

        labels = pred[0].cpu().numpy().astype(np.int32)        # (N,)
        conf_  = conf[0].cpu().numpy()                         # (N,)

        # Low-confidence points → unlabelled (keep in labelled cloud, skip in class clouds)
        # (class 0 = car, so we mark uncertain detections separately via colour)

        # 5. Publish labelled cloud (all points with colour = class)
        header = msg.header
        self._pub_labelled.publish(make_labelled_cloud(header, pts, labels))

        # 6. Publish person cloud
        person_mask = np.isin(labels, list(PERSON_CLASSES)) & (conf_ >= self._score_thresh)
        if person_mask.any():
            self._pub_person.publish(make_class_cloud(header, pts[person_mask]))

        # 7. Publish ground cloud
        ground_mask = np.isin(labels, list(GROUND_CLASSES)) & (conf_ >= self._score_thresh)
        if ground_mask.any():
            self._pub_ground.publish(make_class_cloud(header, pts[ground_mask]))

        # 8. Publish class-coloured sphere markers (one per detected class cluster centroid)
        self._publish_markers(header, pts, labels, conf_)

    def _cluster_by_proximity(self, pts: np.ndarray, radius: float):
        """
        Greedy XY-plane proximity clustering.

        Groups points whose XY distance to the running cluster centroid is within
        `radius` metres.  Returns a list of point-index arrays, one per cluster.
        This is O(N²) but N for person points is small in practice (< a few hundred).
        """
        if len(pts) == 0:
            return []
        remaining = list(range(len(pts)))
        clusters  = []
        while remaining:
            seed = remaining.pop(0)
            members = [seed]
            cx, cy  = pts[seed, 0], pts[seed, 1]
            keep    = []
            for idx in remaining:
                dist = np.hypot(pts[idx, 0] - cx, pts[idx, 1] - cy)
                if dist <= radius:
                    members.append(idx)
                    # Update centroid incrementally
                    cx = pts[members, 0].mean()
                    cy = pts[members, 1].mean()
                else:
                    keep.append(idx)
            remaining = keep
            clusters.append(np.array(members))
        return clusters

    def _publish_markers(self, header, pts, labels, conf):
        markers = MarkerArray()
        mid = 0

        for cls_id in range(self._num_classes):
            mask = (labels == cls_id) & (conf >= self._score_thresh)
            if not mask.any():
                continue
            cls_pts   = pts[mask]
            name, rgb = LABELS[cls_id]
            r, g, b   = [v / 255.0 for v in rgb]

            if cls_id in PERSON_CLASSES:
                # One marker per individual person cluster so the count is visible
                clusters = self._cluster_by_proximity(cls_pts, self._cluster_radius)
                for ci, member_idx in enumerate(clusters):
                    centroid = cls_pts[member_idx].mean(axis=0)
                    m = Marker()
                    m.header              = header
                    m.ns                  = 'semantic'
                    m.id                  = mid
                    m.type                = Marker.SPHERE
                    m.action              = Marker.ADD
                    m.pose.position.x     = float(centroid[0])
                    m.pose.position.y     = float(centroid[1])
                    m.pose.position.z     = float(centroid[2])
                    m.pose.orientation.w  = 1.0
                    m.scale.x = m.scale.y = m.scale.z = 0.5
                    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.9
                    m.lifetime.sec        = 0
                    m.lifetime.nanosec    = int(0.5e9)
                    m.text                = f'{name} #{ci + 1}'
                    markers.markers.append(m)
                    mid += 1
            else:
                # Non-person classes: one centroid marker per class
                centroid = cls_pts.mean(axis=0)
                m = Marker()
                m.header              = header
                m.ns                  = 'semantic'
                m.id                  = mid
                m.type                = Marker.SPHERE
                m.action              = Marker.ADD
                m.pose.position.x     = float(centroid[0])
                m.pose.position.y     = float(centroid[1])
                m.pose.position.z     = float(centroid[2])
                m.pose.orientation.w  = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.4
                m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.8
                m.lifetime.sec        = 0
                m.lifetime.nanosec    = int(0.5e9)
                m.text                = name
                markers.markers.append(m)
                mid += 1

        # Delete stale markers from previous frames
        for i in range(mid, mid + 20):
            m = Marker()
            m.header  = header
            m.ns      = 'semantic'
            m.id      = i
            m.action  = Marker.DELETE
            markers.markers.append(m)

        if mid > 0:
            person_count = sum(
                1 for mk in markers.markers
                if mk.action == Marker.ADD and mk.ns == 'semantic'
                and any(mk.text.startswith(LABELS[c][0]) for c in PERSON_CLASSES)
            )
            if person_count:
                self.get_logger().info(
                    f'Detected {person_count} person(s)', throttle_duration_sec=1.0)

        self._pub_markers.publish(markers)

    def destroy_node(self):
        self._shutdown.set()
        self._worker.join(timeout=2.0)
        super().destroy_node()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = RandLANetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
