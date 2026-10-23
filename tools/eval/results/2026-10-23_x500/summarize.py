"""A3 comparison table from ~/eval_runs/<bag>_<stack>/metrics.json."""
import json, os
R = os.path.expanduser('~/eval_runs')
bags = ['still5', 'loop5', 'height', 'yaw', 'hallway']
cols = ['end_start_gap_m', 'xy_gap_m', 'z_gap_m', 'z_max_dev_m', 'path_m', 'poses']
print(f"{'bag':8s} {'stack':8s} {'gap cm':>7s} {'xy cm':>6s} {'z cm':>6s} {'zmax cm':>8s} {'path m':>7s} {'poses':>6s} {'cpu%':>6s} {'cpu95':>6s} {'rss MB':>7s}")
for b in bags:
    for s in ('upstream', 'fork'):
        f = os.path.join(R, f'{b}_{s}', 'metrics.json')
        if not os.path.exists(f):
            print(f'{b:8s} {s:8s}  (missing)'); continue
        m = json.load(open(f)); t = m['trajectory']; r = m['lio'].get('resources', {})
        print(f"{b:8s} {s:8s} {t['end_start_gap_m']*100:7.1f} {t['xy_gap_m']*100:6.1f} {t['z_gap_m']*100:6.1f} "
              f"{t['z_max_dev_m']*100:8.1f} {t['path_m']:7.1f} {t['poses']:6d} {r.get('cpu_mean_pct', float('nan')):6.0f} "
              f"{r.get('cpu_p95_pct', float('nan')):6.0f} {r.get('rss_max_mb', float('nan')):7.0f}")
