#!/usr/bin/env python3
"""
filter_map_pcd.py — Post-process a saved FastLIO map PCD to remove human-shaped clusters.

Runs the same DBSCAN geometry classifier used by the live dbscan_filter_node on a
static PCD file. Human-shaped clusters (height, footprint, aspect ratio, point count
within configured bounds) are removed and the cleaned cloud is saved.

Usage:
    python3 filter_map_pcd.py map.pcd
    python3 filter_map_pcd.py map.pcd --output map_clean.pcd
    python3 filter_map_pcd.py map.pcd --eps 0.20 --min-samples 3
    python3 filter_map_pcd.py map.pcd --voxel-size 0.05   # downsample before filtering
    python3 filter_map_pcd.py map.pcd --dry-run            # show stats, don't save

Requires: open3d, numpy, scikit-learn
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("Error: open3d not found. Install with:  pip3 install open3d")

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    sys.exit("Error: scikit-learn not found. Install with:  pip3 install scikit-learn")


# ── Geometry classifier (mirrors dbscan_filter_node.py: is_human_cluster) ────

def is_human_cluster(pts, min_h, max_h, min_fp, max_fp, min_pts, max_pts, min_asp):
    if len(pts) < min_pts or len(pts) > max_pts:
        return False
    height = pts[:, 2].max() - pts[:, 2].min()
    if height < min_h or height > max_h:
        return False
    footprint = np.sqrt((pts[:, 0].max() - pts[:, 0].min()) ** 2 +
                        (pts[:, 1].max() - pts[:, 1].min()) ** 2)
    if footprint < min_fp or footprint > max_fp:
        return False
    if footprint > 0 and (height / footprint) < min_asp:
        return False
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Remove human-shaped clusters from a FastLIO map PCD.")
    parser.add_argument("input", help="Input PCD file (e.g. maps/map.pcd)")
    parser.add_argument("--output", "-o",
                        help="Output PCD file. Default: <input>_filtered.pcd")
    parser.add_argument("--voxel-size", type=float, default=0.0,
                        help="Downsample voxel size in metres before filtering (0 = disabled)")

    # DBSCAN
    parser.add_argument("--eps",         type=float, default=0.25,
                        help="DBSCAN cluster radius (m) [default: 0.25]")
    parser.add_argument("--min-samples", type=int,   default=5,
                        help="DBSCAN min points per cluster [default: 5]")

    # Ground strip
    parser.add_argument("--ground-z-min", type=float, default=0.0,
                        help="Strip points below this Z (world frame) [default: 0.0]")
    parser.add_argument("--ground-z-max", type=float, default=3.5,
                        help="Strip points above this Z [default: 3.5]")

    # Human shape thresholds
    parser.add_argument("--min-height",    type=float, default=1.3)
    parser.add_argument("--max-height",    type=float, default=2.2)
    parser.add_argument("--min-footprint", type=float, default=0.15)
    parser.add_argument("--max-footprint", type=float, default=1.0)
    parser.add_argument("--min-pts",       type=int,   default=15)
    parser.add_argument("--max-pts",       type=int,   default=3000)
    parser.add_argument("--min-aspect",    type=float, default=1.8,
                        help="Min height/footprint ratio [default: 1.8]")

    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats only — do not write output file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: file not found: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_filtered.pcd")

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading {input_path} ...")
    t0 = time.time()
    pcd = o3d.io.read_point_cloud(str(input_path))
    pts_full = np.asarray(pcd.points, dtype=np.float32)
    print(f"  Loaded {len(pts_full):,} points  ({time.time()-t0:.1f}s)")

    if len(pts_full) == 0:
        sys.exit("Error: point cloud is empty.")

    # ── Optional voxel downsample ─────────────────────────────────────────────
    if args.voxel_size > 0:
        print(f"Downsampling with voxel size {args.voxel_size} m ...")
        pcd_ds = pcd.voxel_down_sample(args.voxel_size)
        pts_ds = np.asarray(pcd_ds.points, dtype=np.float32)
        print(f"  After downsample: {len(pts_ds):,} points")
    else:
        pts_ds = pts_full

    # ── Ground strip (operate on Z-stripped copy for DBSCAN) ─────────────────
    z_mask = (pts_ds[:, 2] > args.ground_z_min) & (pts_ds[:, 2] < args.ground_z_max)
    pts_work = pts_ds[z_mask]
    print(f"  After Z strip [{args.ground_z_min}, {args.ground_z_max}] m: "
          f"{len(pts_work):,} points")

    if len(pts_work) < args.min_samples:
        print("Too few points after ground strip — nothing to filter.")
        if not args.dry_run:
            o3d.io.write_point_cloud(str(output_path), pcd)
            print(f"Saved unchanged cloud → {output_path}")
        return

    # ── DBSCAN on XY ─────────────────────────────────────────────────────────
    print(f"Running DBSCAN (eps={args.eps}, min_samples={args.min_samples}) "
          f"on {len(pts_work):,} points ...")
    t1 = time.time()
    db = DBSCAN(eps=args.eps, min_samples=args.min_samples, n_jobs=-1).fit(pts_work[:, :2])
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  Found {n_clusters} clusters  ({time.time()-t1:.1f}s)")

    # ── Classify clusters ─────────────────────────────────────────────────────
    human_xy_centers = []
    human_clusters = 0
    human_mask = np.zeros(len(pts_work), dtype=bool)

    for lbl in set(labels) - {-1}:
        c = pts_work[labels == lbl]
        if is_human_cluster(c,
                            args.min_height,    args.max_height,
                            args.min_footprint, args.max_footprint,
                            args.min_pts,       args.max_pts,
                            args.min_aspect):
            human_mask[labels == lbl] = True
            human_xy_centers.append((c[:, 0].mean(), c[:, 1].mean()))
            human_clusters += 1

    n_stripped = human_mask.sum()
    n_kept_work = (~human_mask).sum()
    print(f"  Human clusters: {human_clusters} | "
          f"Points stripped: {n_stripped:,} | Points kept: {n_kept_work:,}")

    if human_clusters > 0:
        print("  Human cluster locations (XY world frame):")
        for i, (cx, cy) in enumerate(human_xy_centers):
            print(f"    [{i+1}]  x={cx:.2f}  y={cy:.2f}")

    # ── Build indices to KEEP in the full cloud ───────────────────────────────
    # human_mask is over pts_work (z-stripped + downsampled).
    # We need to remove the corresponding points from pts_ds using XY proximity
    # to the detected human cluster centroids, then re-map to the original cloud.
    #
    # Strategy: for each human centroid, mark all original points within eps radius
    # in XY AND within the Z range as belonging to a human cluster.

    r_sq = (args.eps * 1.5) ** 2  # slightly wider radius for full-density original cloud

    if human_clusters == 0:
        print("No human clusters found — output will be identical to input.")
        keep_mask_full = np.ones(len(pts_full), dtype=bool)
    else:
        keep_mask_full = np.ones(len(pts_full), dtype=bool)
        for cx, cy in human_xy_centers:
            # XY distance to centroid on the FULL original cloud
            dist2 = (pts_full[:, 0] - cx) ** 2 + (pts_full[:, 1] - cy) ** 2
            # Also apply Z filter so we don't remove ground/ceiling near the person
            in_z = (pts_full[:, 2] > args.ground_z_min) & (pts_full[:, 2] < args.ground_z_max)
            keep_mask_full &= ~(in_z & (dist2 < r_sq))

    n_final = keep_mask_full.sum()
    n_removed_total = len(pts_full) - n_final
    pct = 100.0 * n_removed_total / len(pts_full) if len(pts_full) > 0 else 0

    print(f"\nSummary:")
    print(f"  Input points   : {len(pts_full):>10,}")
    print(f"  Removed points : {n_removed_total:>10,}  ({pct:.2f}%)")
    print(f"  Output points  : {n_final:>10,}")

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    # ── Save ─────────────────────────────────────────────────────────────────
    pts_out = pts_full[keep_mask_full]
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(pts_out)

    # Preserve colours if present
    if pcd.has_colors():
        colours = np.asarray(pcd.colors)
        pcd_out.colors = o3d.utility.Vector3dVector(colours[keep_mask_full])

    print(f"\nSaving → {output_path} ...")
    o3d.io.write_point_cloud(str(output_path), pcd_out)
    print("Done.")


if __name__ == "__main__":
    main()
