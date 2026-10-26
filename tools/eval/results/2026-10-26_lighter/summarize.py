"""Lighter-lio same-day table from ~/eval_runs/lighter/<variant>_<bag>/ (settled gap as in summarize.py)."""
import csv, json, math, os, subprocess
R = os.path.expanduser('~/eval_runs/lighter')
E = os.path.expanduser('~/fastlio2_ws/src/FASTLIO2_ROS2/tools/eval/eval.py')
print(f"{'bag':8s} {'variant':8s} {'settled cm':>10s} {'gap cm':>7s} {'zmax cm':>8s} {'path m':>7s} {'poses':>6s} "
      f"{'cpu%':>5s} {'cpu95':>6s} {'rss MB':>7s} {'vs base rms/max mm':>19s}")
for b in ['still5', 'loop5', 'height', 'yaw', 'hallway']:
    for v in ('upstream', 'base', 'r1', 'l4'):
        d = os.path.join(R, f'{v}_{b}')
        if not os.path.exists(os.path.join(d, 'metrics.json')):
            print(f'{b:8s} {v:8s}  (missing)'); continue
        m = json.load(open(os.path.join(d, 'metrics.json'))); t = m['trajectory']; r = m['lio'].get('resources', {})
        rows = list(csv.DictReader(open(os.path.join(d, 'trajectory.csv'))))
        T = [float(x['t']) for x in rows]; P = [(float(x['x']), float(x['y']), float(x['z'])) for x in rows]
        i = next(j for j, x in enumerate(T) if x - T[0] >= 8.0); a, e = P[i], P[-1]
        cmp = ''
        if v in ('r1', 'l4'):
            out = subprocess.run(['python3', E, 'compare', os.path.join(R, f'base_{b}'), d], capture_output=True, text=True).stdout
            j = json.loads(out[out.index('{'):])
            cmp = f"{j['rms_mm']:8.2f}/{j['max_mm']:6.1f}"
        print(f"{b:8s} {v:8s} {math.dist(a, e)*100:10.1f} {t['end_start_gap_m']*100:7.1f} {t['z_max_dev_m']*100:8.1f} "
              f"{t['path_m']:7.1f} {t['poses']:6d} {r.get('cpu_mean_pct', float('nan')):5.0f} "
              f"{r.get('cpu_p95_pct', float('nan')):6.0f} {r.get('rss_max_mb', float('nan')):7.0f} {cmp:>19s}")
    print()
