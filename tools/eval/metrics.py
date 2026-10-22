"""Metrics for replay evaluation runs. Pure numpy, no ROS, so they can be unit-tested.

Inputs are plain arrays:
  trajectory: t (N,), p (N,3) positions, q (N,4) quaternions x,y,z,w
  status:     list of (t, state, dead_reckoning) from /localizer/localizer_status
  offset:     t (M,), p (M,3) translation of the map -> local frame transform
"""
import numpy as np


def trajectory_metrics(t, p):
    """Path length, furthest distance from the start, and the end-to-start gap.

    On a recording that returns to its starting mark, the gap is the drift.
    """
    p = np.asarray(p, dtype=float)
    if len(p) < 2:
        return {'poses': int(len(p)), 'duration_s': 0.0, 'path_m': 0.0,
                'max_from_start_m': 0.0, 'end_start_gap_m': 0.0}
    steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
    from_start = np.linalg.norm(p - p[0], axis=1)
    return {
        'poses': int(len(p)),
        'duration_s': float(t[-1] - t[0]),
        'path_m': float(steps.sum()),
        'max_from_start_m': float(from_start.max()),
        'end_start_gap_m': float(from_start[-1]),
    }


def compare_trajectories(ta, pa, qa, tb, pb, qb, decimals=4):
    """Difference between two runs of the same recording, on matched stamps."""
    common, ia, ib = np.intersect1d(np.round(ta, decimals), np.round(tb, decimals),
                                    return_indices=True)
    if len(common) == 0:
        return {'matched': 0}
    dp = np.linalg.norm(np.asarray(pa)[ia] - np.asarray(pb)[ib], axis=1)
    dot = np.abs(np.sum(np.asarray(qa)[ia] * np.asarray(qb)[ib], axis=1)).clip(0.0, 1.0)
    rot_deg = np.degrees(2.0 * np.arccos(dot))
    return {
        'matched': int(len(common)),
        'rms_mm': float(np.sqrt(np.mean(dp ** 2)) * 1000.0),
        'max_mm': float(dp.max() * 1000.0),
        'max_rot_deg': float(rot_deg.max()),
    }


def localizer_metrics(status, offset_t=None, offset_p=None, jump_m=0.2):
    """Localizer events from the status stream and the map->local offset.

    status: (t, state, dead_reckoning) tuples in time order.
    A loss is a change from TRACKING to any other state; its recovery time runs
    until the next TRACKING. A jump is a change of the map->local offset larger
    than jump_m between consecutive TF messages (a relocalization moving the map).
    """
    status = list(status)
    out = {'status_msgs': len(status), 'time_to_first_fix_s': None, 'losses': 0,
           'recovery_s': [], 'unrecovered': 0, 'tracking_fraction': 0.0,
           'dead_reckoning_s': 0.0, 'offset_jumps': 0, 'max_offset_jump_m': 0.0}
    if status:
        t0 = status[0][0]
        lost_at = None
        tracking_time = dr_time = 0.0
        for i, (t, state, dr) in enumerate(status):
            if state == 'TRACKING':
                if out['time_to_first_fix_s'] is None:
                    out['time_to_first_fix_s'] = float(t - t0)
                if lost_at is not None:
                    out['recovery_s'].append(float(t - lost_at))
                    lost_at = None
            elif i > 0 and status[i - 1][1] == 'TRACKING':
                out['losses'] += 1
                lost_at = t
            if i + 1 < len(status):
                dt = status[i + 1][0] - t
                tracking_time += dt if state == 'TRACKING' else 0.0
                dr_time += dt if dr else 0.0
        span = status[-1][0] - t0
        out['tracking_fraction'] = float(tracking_time / span) if span > 0 else 0.0
        out['dead_reckoning_s'] = float(dr_time)
        out['unrecovered'] = 1 if lost_at is not None else 0
    if offset_p is not None and len(offset_p) > 1:
        jumps = np.linalg.norm(np.diff(np.asarray(offset_p, dtype=float), axis=0), axis=1)
        out['offset_jumps'] = int((jumps > jump_m).sum())
        out['max_offset_jump_m'] = float(jumps.max())
    return out


def parse_status(text):
    """'mode=GLOBAL state=TRACKING dead_reckoning=0 ...' -> (state, dead_reckoning)."""
    fields = dict(kv.split('=', 1) for kv in text.split() if '=' in kv)
    return fields.get('state', ''), fields.get('dead_reckoning', '0') not in ('0', 'false')
