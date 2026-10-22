import numpy as np
import pytest

import metrics


def square_loop(n_side=10, side=2.0, gap=(0.0, 0.0)):
    """Walk a side x side square and stop `gap` metres short of the start."""
    corners = np.array([[0, 0], [side, 0], [side, side], [0, side], [0, 0]], dtype=float)
    pts = [corners[i] + (corners[i + 1] - corners[i]) * k / n_side
           for i in range(4) for k in range(n_side)]
    pts.append(corners[-1] + np.array(gap))
    p = np.column_stack([np.array(pts), np.zeros(len(pts))])
    return np.arange(len(p)) * 0.1, p


def test_closed_loop_has_no_gap():
    t, p = square_loop()
    m = metrics.trajectory_metrics(t, p)
    assert m['path_m'] == pytest.approx(8.0)
    assert m['end_start_gap_m'] == pytest.approx(0.0)
    assert m['max_from_start_m'] == pytest.approx(np.hypot(2.0, 2.0))


def test_loop_gap_is_the_drift():
    t, p = square_loop(gap=(0.03, 0.04))
    assert metrics.trajectory_metrics(t, p)['end_start_gap_m'] == pytest.approx(0.05)


def test_identical_runs_compare_to_zero():
    t, p = square_loop()
    q = np.tile([0.0, 0.0, 0.0, 1.0], (len(t), 1))
    c = metrics.compare_trajectories(t, p, q, t, p, q)
    assert c['matched'] == len(t)
    assert c['rms_mm'] == 0.0 and c['max_rot_deg'] == 0.0


def test_compare_uses_matched_stamps_only():
    t, p = square_loop()
    q = np.tile([0.0, 0.0, 0.0, 1.0], (len(t), 1))
    shifted = p + [0.0, 0.0, 0.01]
    c = metrics.compare_trajectories(t, p, q, t[5:], shifted[5:], q[5:])
    assert c['matched'] == len(t) - 5
    assert c['max_mm'] == pytest.approx(10.0)


def test_rotation_difference():
    t = np.array([0.0])
    half = np.radians(10.0) / 2
    qa = np.array([[0, 0, 0, 1.0]])
    qb = np.array([[0, 0, np.sin(half), np.cos(half)]])
    c = metrics.compare_trajectories(t, np.zeros((1, 3)), qa, t, np.zeros((1, 3)), qb)
    assert c['max_rot_deg'] == pytest.approx(10.0)


def test_localizer_events():
    s = [(0, 'IDLE', False), (1, 'SEARCHING', False), (3, 'TRACKING', False),
         (10, 'TRACKING', False), (11, 'IDLE', False), (12, 'SEARCHING', False),
         (14, 'TRACKING', False), (20, 'TRACKING', False)]
    m = metrics.localizer_metrics(s)
    assert m['time_to_first_fix_s'] == 3.0
    assert m['losses'] == 1
    assert m['recovery_s'] == [3.0]
    assert m['unrecovered'] == 0
    assert m['tracking_fraction'] == pytest.approx((8 + 6) / 20)


def test_unrecovered_loss_and_dead_reckoning():
    s = [(0, 'TRACKING', False), (2, 'IDLE', False), (3, 'SEARCHING', False),
         (5, 'TRACKING', True), (9, 'TRACKING', True)]
    m = metrics.localizer_metrics(s)
    assert m['losses'] == 1 and m['recovery_s'] == [3.0]
    assert m['dead_reckoning_s'] == pytest.approx(4.0)
    s2 = [(0, 'TRACKING', False), (2, 'IDLE', False), (3, 'SEARCHING', False)]
    assert metrics.localizer_metrics(s2)['unrecovered'] == 1


def test_offset_jumps():
    off = np.array([[0, 0, 0], [0.01, 0, 0], [0.5, 0, 0], [0.5, 0.02, 0]], dtype=float)
    m = metrics.localizer_metrics([], np.arange(4.0), off, jump_m=0.2)
    assert m['offset_jumps'] == 1
    assert m['max_offset_jump_m'] == pytest.approx(0.49)


def test_parse_status():
    assert metrics.parse_status('mode=GLOBAL state=TRACKING dead_reckoning=1 mapping=0') == ('TRACKING', True)
    assert metrics.parse_status('mode=ICP state=IDLE dead_reckoning=0') == ('IDLE', False)
