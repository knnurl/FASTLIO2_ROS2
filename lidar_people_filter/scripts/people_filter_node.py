#!/usr/bin/env python3
"""
LiDAR People Detection and Removal Node for ROS2
-------------------------------------------------
Detects humans in 3D LiDAR point clouds and removes them
before publishing to downstream 3D mapping pipelines.

Pipeline:
  PointCloud2 → Voxel Downsample → Ground Removal → Euclidean Clustering
              → Human Classification → Point Removal → Filtered PointCloud2

Author: lidar_people_filter package
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point

import numpy as np
import struct
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    import warnings
    warnings.warn("scikit-learn not found; falling back to simple Euclidean clustering.")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClusterFeatures:
    """Geometric features extracted from a point cluster."""
    centroid: np.ndarray          # (3,) XYZ centroid
    min_bound: np.ndarray         # (3,) bounding box min
    max_bound: np.ndarray         # (3,) bounding box max
    width: float                  # X extent (metres)
    depth: float                  # Y extent (metres)
    height: float                 # Z extent (metres)
    num_points: int
    point_density: float          # points / volume
    aspect_ratio_wh: float        # width / height
    aspect_ratio_wd: float        # width / depth
    ground_z: float               # lowest Z in cluster


# ─────────────────────────────────────────────────────────────────────────────
# PointCloud2 helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def pc2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Extract (N, 3) float32 XYZ array from PointCloud2, ignoring NaNs."""
    # Build a numpy structured dtype from the message fields
    offsets = {f.name: f.offset for f in msg.fields}
    dtypes  = {f.name: DTYPE_MAP.get(f.datatype, np.float32) for f in msg.fields}

    point_step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8)
    n_points = msg.width * msg.height

    # Fast path: packed XYZI or XYZ float32
    has_xyz = all(k in offsets for k in ('x', 'y', 'z'))
    if not has_xyz:
        raise ValueError("PointCloud2 message has no x/y/z fields")

    x = np.frombuffer(data[offsets['x']:n_points * point_step:point_step].tobytes()
                      if False else data, dtype=np.uint8)

    # Structured array approach (handles any layout)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    raw = raw[:n_points * point_step].reshape(n_points, point_step)

    def read_field(name):
        off  = offsets[name]
        dt   = dtypes[name]
        size = np.dtype(dt).itemsize
        return np.frombuffer(raw[:, off:off + size].tobytes(), dtype=dt)

    xyz = np.stack([read_field('x'), read_field('y'), read_field('z')], axis=1).astype(np.float32)
    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid], np.where(valid)[0]  # points, original indices


def xyz_to_pc2(header: Header, points: np.ndarray) -> PointCloud2:
    """Create a PointCloud2 message from an (N, 3) float32 array."""
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width  = len(points)
    msg.is_dense = True
    msg.is_bigendian = False
    msg.point_step = 12   # 3 × float32
    msg.row_step   = msg.point_step * msg.width

    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = points.astype(np.float32).tobytes()
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# Processing pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Simple voxel-grid downsample (one point per voxel, fastest non-lib approach)."""
    if voxel_size <= 0 or len(points) == 0:
        return points
    shifted = points - points.min(axis=0)
    voxel_indices = (shifted / voxel_size).astype(np.int32)
    # Hash voxel (ix, iy, iz) → keep first encountered point
    keys = voxel_indices[:, 0] * 10_000_000 + voxel_indices[:, 1] * 10_000 + voxel_indices[:, 2]
    _, first = np.unique(keys, return_index=True)
    return points[first]


def remove_ground_ransac(
    points: np.ndarray,
    distance_thresh: float = 0.15,
    max_iterations: int = 100,
    min_inlier_ratio: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC plane fitting to remove the ground.

    Returns (non_ground_points, ground_points).
    """
    if len(points) < 3:
        return points, np.empty((0, 3), dtype=np.float32)

    best_inliers = np.array([], dtype=np.int32)

    rng = np.random.default_rng(42)
    for _ in range(max_iterations):
        sample = rng.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[sample]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        d = -normal.dot(p1)
        distances = np.abs(points @ normal + d)
        inliers = np.where(distances < distance_thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            if len(inliers) / len(points) > min_inlier_ratio:
                break

    mask = np.ones(len(points), dtype=bool)
    mask[best_inliers] = False
    return points[mask], points[best_inliers]


def euclidean_cluster_simple(
    points: np.ndarray,
    tolerance: float,
    min_pts: int,
    max_pts: int,
) -> List[np.ndarray]:
    """
    Naive Euclidean clustering fallback (O(N²) — fine for small clouds).
    Returns list of point arrays.
    """
    if len(points) == 0:
        return []

    visited = np.zeros(len(points), dtype=bool)
    clusters = []

    for i in range(len(points)):
        if visited[i]:
            continue
        dists = np.linalg.norm(points - points[i], axis=1)
        neighbours = np.where(dists <= tolerance)[0]
        if len(neighbours) < min_pts:
            visited[i] = True
            continue
        cluster_idx = list(neighbours)
        visited[neighbours] = True
        j = 0
        while j < len(cluster_idx):
            idx = cluster_idx[j]
            d2 = np.linalg.norm(points - points[idx], axis=1)
            new_nb = np.where((d2 <= tolerance) & ~visited)[0]
            visited[new_nb] = True
            cluster_idx.extend(new_nb.tolist())
            j += 1
        if min_pts <= len(cluster_idx) <= max_pts:
            clusters.append(points[np.array(cluster_idx)])

    return clusters


def cluster_points(
    points: np.ndarray,
    tolerance: float,
    min_pts: int,
    max_pts: int,
    use_dbscan: bool = True,
) -> List[np.ndarray]:
    """Cluster non-ground points. Uses DBSCAN if sklearn is available."""
    if len(points) == 0:
        return []

    if use_dbscan and SKLEARN_AVAILABLE:
        labels = DBSCAN(eps=tolerance, min_samples=min_pts, n_jobs=-1).fit_predict(points[:, :2])
        clusters = []
        for lbl in np.unique(labels):
            if lbl < 0:
                continue  # noise
            mask = labels == lbl
            if mask.sum() <= max_pts:
                clusters.append(points[mask])
        return clusters
    else:
        return euclidean_cluster_simple(points, tolerance, min_pts, max_pts)


def extract_features(cluster: np.ndarray) -> ClusterFeatures:
    """Compute geometric features for a single cluster."""
    min_b = cluster.min(axis=0)
    max_b = cluster.max(axis=0)
    extent = max_b - min_b + 1e-6
    volume = max(extent[0] * extent[1] * extent[2], 1e-6)
    return ClusterFeatures(
        centroid       = cluster.mean(axis=0),
        min_bound      = min_b,
        max_bound      = max_b,
        width          = float(extent[0]),
        depth          = float(extent[1]),
        height         = float(extent[2]),
        num_points     = len(cluster),
        point_density  = len(cluster) / volume,
        aspect_ratio_wh= float(extent[0] / extent[2]),
        aspect_ratio_wd= float(extent[0] / extent[1]),
        ground_z       = float(min_b[2]),
    )


def classify_human(feat: ClusterFeatures, params: dict) -> bool:
    """
    Rule-based human classifier using bounding-box geometry.

    A cluster is considered human if ALL of the following hold:
      - Height within [min_height, max_height]
      - Width within [min_width, max_width]
      - Depth within [min_depth, max_depth]
      - Horizontal cross-section is roughly square (not elongated)
      - Point count within expected range
    """
    h_ok   = params['min_height'] <= feat.height <= params['max_height']
    w_ok   = params['min_width']  <= feat.width  <= params['max_width']
    d_ok   = params['min_depth']  <= feat.depth  <= params['max_depth']
    pts_ok = params['min_points'] <= feat.num_points <= params['max_points']

    # Horizontal aspect ratio: width/depth should be < max_horizontal_aspect
    # (a person cross-section is roughly circular)
    h_aspect_ok = feat.aspect_ratio_wd < params['max_horizontal_aspect']

    # Height-to-width ratio: people are tall and narrow
    vert_aspect_ok = feat.aspect_ratio_wh < params['max_vertical_aspect']

    return h_ok and w_ok and d_ok and pts_ok and h_aspect_ok and vert_aspect_ok


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────────────────────

class PeopleFilterNode(Node):

    def __init__(self):
        super().__init__('people_filter_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameters('', [
            # I/O
            ('input_topic',  '/velodyne_points'),
            ('output_topic', '/points_no_people'),
            ('marker_topic', '/people_detections'),
            ('publish_markers', True),

            # Voxel downsampling
            ('voxel_size', 0.05),           # metres

            # Ground removal
            ('ground_distance_thresh', 0.15),
            ('ground_max_iterations',  100),

            # Clustering
            ('cluster_tolerance', 0.5),     # DBSCAN eps / Euclidean radius
            ('cluster_min_points', 10),
            ('cluster_max_points', 2500),
            ('use_dbscan', True),

            # Human geometry thresholds
            ('human_min_height', 1.2),
            ('human_max_height', 2.2),
            ('human_min_width',  0.2),
            ('human_max_width',  1.2),
            ('human_min_depth',  0.2),
            ('human_max_depth',  1.2),
            ('human_min_points', 10),
            ('human_max_points', 2000),
            ('human_max_horizontal_aspect', 4.0),
            ('human_max_vertical_aspect',   1.5),

            # Bounding-box inflation for removal (metres added each side)
            ('removal_inflation', 0.2),
        ])

        p = self.get_param

        # ── Publishers / Subscribers ───────────────────────────────────────
        self._pub_filtered = self.create_publisher(
            PointCloud2, p('output_topic'), 10)
        self._pub_markers = self.create_publisher(
            MarkerArray, p('marker_topic'), 10) if p('publish_markers') else None

        self._sub = self.create_subscription(
            PointCloud2, p('input_topic'), self._callback, 10)

        self.get_logger().info(
            f"PeopleFilterNode ready\n"
            f"  input  → {p('input_topic')}\n"
            f"  output → {p('output_topic')}\n"
            f"  DBSCAN → {p('use_dbscan')} (sklearn={SKLEARN_AVAILABLE})"
        )

    # ── Parameter helper ───────────────────────────────────────────────────

    def get_param(self, name):
        return self.get_parameter(name).value

    # ── Main callback ──────────────────────────────────────────────────────

    def _callback(self, msg: PointCloud2):
        t0 = time.perf_counter()
        p  = self.get_param

        # 1. Deserialise
        try:
            xyz, _valid_idx = pc2_to_xyz(msg)
        except Exception as exc:
            self.get_logger().error(f"Failed to parse PointCloud2: {exc}")
            return

        if len(xyz) == 0:
            self._pub_filtered.publish(xyz_to_pc2(msg.header, xyz))
            return

        # 2. Voxel downsample
        xyz_ds = voxel_downsample(xyz, p('voxel_size'))

        # 3. Ground removal
        non_ground, _ground = remove_ground_ransac(
            xyz_ds,
            distance_thresh=p('ground_distance_thresh'),
            max_iterations=p('ground_max_iterations'),
        )

        # 4. Cluster non-ground points
        clusters = cluster_points(
            non_ground,
            tolerance=p('cluster_tolerance'),
            min_pts=p('cluster_min_points'),
            max_pts=p('cluster_max_points'),
            use_dbscan=p('use_dbscan'),
        )

        # 5. Classify clusters → collect human bounding boxes
        human_params = {
            'min_height':          p('human_min_height'),
            'max_height':          p('human_max_height'),
            'min_width':           p('human_min_width'),
            'max_width':           p('human_max_width'),
            'min_depth':           p('human_min_depth'),
            'max_depth':           p('human_max_depth'),
            'min_points':          p('human_min_points'),
            'max_points':          p('human_max_points'),
            'max_horizontal_aspect': p('human_max_horizontal_aspect'),
            'max_vertical_aspect':   p('human_max_vertical_aspect'),
        }
        inf = p('removal_inflation')

        human_boxes  = []   # list of (min_b - inf, max_b + inf)
        human_feats  = []

        for cl in clusters:
            feat = extract_features(cl)
            if classify_human(feat, human_params):
                human_boxes.append((feat.min_bound - inf, feat.max_bound + inf))
                human_feats.append(feat)

        # 6. Remove points inside any human bounding box (applied to full cloud)
        filtered = self._remove_human_points(xyz, human_boxes)

        # 7. Publish
        out_msg = xyz_to_pc2(msg.header, filtered)
        self._pub_filtered.publish(out_msg)

        if self._pub_markers and human_feats:
            self._pub_markers.publish(
                self._build_markers(msg.header, human_feats))

        dt = (time.perf_counter() - t0) * 1000
        self.get_logger().debug(
            f"Processed {len(xyz)} pts → {len(filtered)} pts | "
            f"{len(human_feats)} person(s) removed | {dt:.1f} ms"
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _remove_human_points(
        points: np.ndarray,
        boxes: List[Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Return subset of *points* not inside any human bounding box."""
        if not boxes:
            return points
        keep = np.ones(len(points), dtype=bool)
        for (mn, mx) in boxes:
            inside = (
                (points[:, 0] >= mn[0]) & (points[:, 0] <= mx[0]) &
                (points[:, 1] >= mn[1]) & (points[:, 1] <= mx[1]) &
                (points[:, 2] >= mn[2]) & (points[:, 2] <= mx[2])
            )
            keep &= ~inside
        return points[keep]

    @staticmethod
    def _build_markers(header: Header, feats: List[ClusterFeatures]) -> MarkerArray:
        """Build RViz bounding-box markers for detected people."""
        arr = MarkerArray()

        # Clear old markers
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        for i, feat in enumerate(feats):
            # Box
            m = Marker()
            m.header  = header
            m.ns      = 'people_bbox'
            m.id      = i
            m.type    = Marker.CUBE
            m.action  = Marker.ADD
            m.pose.position.x = float(feat.centroid[0])
            m.pose.position.y = float(feat.centroid[1])
            m.pose.position.z = float(feat.centroid[2])
            m.pose.orientation.w = 1.0
            m.scale.x = max(feat.width,  0.1)
            m.scale.y = max(feat.depth,  0.1)
            m.scale.z = max(feat.height, 0.1)
            m.color   = ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.4)
            m.lifetime.sec = 1
            arr.markers.append(m)

            # Text label
            t = Marker()
            t.header  = header
            t.ns      = 'people_label'
            t.id      = i
            t.type    = Marker.TEXT_VIEW_FACING
            t.action  = Marker.ADD
            t.pose.position.x = float(feat.centroid[0])
            t.pose.position.y = float(feat.centroid[1])
            t.pose.position.z = float(feat.max_bound[2]) + 0.2
            t.pose.orientation.w = 1.0
            t.scale.z = 0.3
            t.color   = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            t.text    = f"Person {i}\n{feat.height:.1f}m"
            t.lifetime.sec = 1
            arr.markers.append(t)

        return arr


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PeopleFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
