#!/usr/bin/env python3
"""
test_ate_rpe.py
───────────────
Feed a known trajectory with injected drift into map_validator,
verify ATE and RPE outputs are within expected bounds.

Tests the mathematical correctness of ATE/RPE computation
using a Python reimplementation.
"""
import unittest
import math
import numpy as np


def horn_alignment(src, dst):
    """Horn's closed-form SE3 alignment (translation only, rotation + translation)."""
    n = min(len(src), len(dst))
    if n < 3:
        return np.eye(4)

    src = np.array(src[:n])
    dst = np.array(dst[:n])

    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)

    H = (src - centroid_src).T @ (dst - centroid_dst)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_dst - R @ centroid_src

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def compute_ate_rmse(estimated, ground_truth):
    """Compute ATE RMSE after Horn alignment."""
    n = min(len(estimated), len(ground_truth))
    if n < 3:
        return 0.0

    src = [e[:3] for e in estimated[:n]]
    dst = [g[:3] for g in ground_truth[:n]]

    T = horn_alignment(src, dst)
    R = T[:3, :3]
    t = T[:3, 3]

    sum_sq = 0.0
    for i in range(n):
        aligned = R @ np.array(src[i]) + t
        sum_sq += np.sum((aligned - np.array(dst[i])) ** 2)

    return math.sqrt(sum_sq / n)


def compute_rpe_translation(trajectory, delta_m=1.0):
    """Compute mean RPE translation error."""
    errors = []
    for i in range(len(trajectory)):
        acc_dist = 0.0
        j = i + 1
        while j < len(trajectory):
            acc_dist += np.linalg.norm(
                np.array(trajectory[j][:3]) - np.array(trajectory[j-1][:3]))
            if acc_dist >= delta_m:
                break
            j += 1
        if j >= len(trajectory):
            break

        t_rel = np.array(trajectory[j][:3]) - np.array(trajectory[i][:3])
        trans_err = abs(np.linalg.norm(t_rel) - acc_dist)
        errors.append(trans_err)

    return np.mean(errors) if errors else 0.0


class TestATERPE(unittest.TestCase):

    def test_perfect_trajectory(self):
        """ATE should be zero for identical trajectories."""
        traj = [(i * 0.1, 0.0, 0.0) for i in range(100)]
        ate = compute_ate_rmse(traj, traj)
        self.assertAlmostEqual(ate, 0.0, places=6)

    def test_constant_offset(self):
        """ATE should be zero for a constant translation offset (Horn aligns it)."""
        estimated = [(i * 0.1, 0.0, 0.0) for i in range(100)]
        ground_truth = [(i * 0.1 + 5.0, 3.0, 0.0) for i in range(100)]
        ate = compute_ate_rmse(estimated, ground_truth)
        self.assertAlmostEqual(ate, 0.0, places=4)

    def test_known_drift(self):
        """ATE should reflect known linear drift."""
        n = 100
        estimated = [(i * 0.1, 0.0, 0.0) for i in range(n)]
        # Add linearly increasing drift in y
        drift_per_step = 0.01  # 1cm per step
        ground_truth = [(i * 0.1, i * drift_per_step, 0.0) for i in range(n)]

        ate = compute_ate_rmse(estimated, ground_truth)
        # After Horn alignment, residual error should be less than the max drift
        max_drift = (n - 1) * drift_per_step
        self.assertLess(ate, max_drift)
        # But greater than zero (drift is not purely translational constant)
        self.assertGreater(ate, 0.0)

    def test_rpe_straight_line(self):
        """RPE should be near zero for a straight-line trajectory."""
        traj = [(i * 0.1, 0.0, 0.0) for i in range(100)]
        rpe = compute_rpe_translation(traj, delta_m=1.0)
        self.assertAlmostEqual(rpe, 0.0, places=4)

    def test_rpe_with_noise(self):
        """RPE should be non-zero for a noisy trajectory."""
        np.random.seed(42)
        traj = [(i * 0.1 + np.random.normal(0, 0.01),
                 np.random.normal(0, 0.01),
                 0.0) for i in range(200)]
        rpe = compute_rpe_translation(traj, delta_m=1.0)
        self.assertGreater(rpe, 0.0)
        # But should be small relative to the step size
        self.assertLess(rpe, 0.1)

    def test_rpe_large_drift(self):
        """RPE should detect large drift."""
        # Trajectory that curves significantly
        traj = [(i * 0.1, 0.01 * i * i * 0.01, 0.0) for i in range(200)]
        rpe = compute_rpe_translation(traj, delta_m=1.0)
        self.assertGreater(rpe, 0.0)

    def test_horn_alignment_rotation(self):
        """Horn's method should handle pure rotation."""
        # Rotate ground truth by 90 degrees
        angle = math.pi / 4  # 45 degrees
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        estimated = [(i * 0.1, 0.0, 0.0) for i in range(50)]
        ground_truth = [(i * 0.1 * cos_a, i * 0.1 * sin_a, 0.0) for i in range(50)]

        ate = compute_ate_rmse(estimated, ground_truth)
        # After alignment, error should be essentially zero
        self.assertAlmostEqual(ate, 0.0, places=4)

    def test_minimum_poses(self):
        """ATE requires at least 3 poses."""
        traj = [(0, 0, 0), (1, 0, 0)]
        ate = compute_ate_rmse(traj, traj)
        self.assertEqual(ate, 0.0)  # Returns 0 for < 3 poses


if __name__ == '__main__':
    unittest.main()
