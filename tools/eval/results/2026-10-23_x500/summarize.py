"""A3 comparison table from ~/eval_runs/<bag>_<stack>/metrics.json and trajectory.csv.

gap = eval.py's end_start_gap_m (last pose vs the very FIRST pose, which is taken before the
filter settles). settled = last pose vs the pose 8 s in, still inside every recording's opening
10 s still period; use this one to compare stacks.
"""
import csv, json, math, os
R = os.path.expanduser('~/eval_runs')
print(f"{'bag':8s} {'stack':8s} {'settled cm':>10s} {'xy':>5s} {'z':>5s} {'gap cm':>7s} {'zmax cm':>8s} "
      f"{'path m':>7s} {'poses':>6s} {'cpu%':>5s} {'cpu95':>6s} {'rss MB':>7s}")
for b in ['still5', 'loop5', 'height', 'yaw', 'hallway']:
    for s in ('upstream', 'fork'):
        d = os.path.join(R, f'{b}_{s}')
        if not os.path.exists(os.path.join(d, 'metrics.json')):
            print(f'{b:8s} {s:8s}  (missing)'); continue
        m = json.load(open(os.path.join(d, 'metrics.json'))); t = m['trajectory']; r = m['lio'].get('resources', {})
        rows = list(csv.DictReader(open(os.path.join(d, 'trajectory.csv'))))
        T = [float(x['t']) for x in rows]; P = [(float(x['x']), float(x['y']), float(x['z'])) for x in rows]
        i = next(j for j, v in enumerate(T) if v - T[0] >= 8.0); a, e = P[i], P[-1]
        print(f"{b:8s} {s:8s} {math.dist(a, e)*100:10.1f} {math.hypot(e[0]-a[0], e[1]-a[1])*100:5.1f} "
              f"{(e[2]-a[2])*100:5.1f} {t['end_start_gap_m']*100:7.1f} {t['z_max_dev_m']*100:8.1f} {t['path_m']:7.1f} "
              f"{t['poses']:6d} {r.get('cpu_mean_pct', float('nan')):5.0f} {r.get('cpu_p95_pct', float('nan')):6.0f} "
              f"{r.get('rss_max_mb', float('nan')):7.0f}")
