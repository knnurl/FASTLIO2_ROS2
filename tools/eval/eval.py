#!/usr/bin/env python3
"""Replay a recording through LIO (and the localizer) and measure the result.

  eval.py run --bag ~/lidar_rec/3 --out runs/bag3            # LIO only
  eval.py run --bag ~/lidar_rec/1 --out runs/bag1 --save-map maps/bag1.pcd
  eval.py run --bag ~/lidar_rec/3 --map maps/map.pcd --out runs/bag3_loc
  eval.py compare runs/before runs/after

Needs the workspace sourced. The run uses its own ROS domain, localhost only,
and ignores CYCLONEDDS_URI, so it cannot talk to anything else on the machine.
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402


def isolate(domain):
    os.environ['ROS_DOMAIN_ID'] = str(domain)
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    os.environ.pop('CYCLONEDDS_URI', None)   # a URI selecting lo clashes with LOCALHOST_ONLY


def share(pkg, *parts):
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory(pkg), *parts)


def start(cmd, log):
    return subprocess.Popen(cmd, stdout=open(log, 'w'), stderr=subprocess.STDOUT,
                            start_new_session=True)


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


class Recorder:
    """Records LIO odometry, localizer status and the map->local offset to CSV."""

    def __init__(self, out, map_frame, odom_topic):
        import rclpy
        from nav_msgs.msg import Odometry
        from std_msgs.msg import String
        from tf2_msgs.msg import TFMessage
        rclpy.init()
        self.rclpy = rclpy
        self.node = rclpy.create_node('eval_recorder')
        self.files = {k: open(out / f'{k}.csv', 'w', newline='') for k in ('trajectory', 'status', 'offset')}
        self.w = {k: csv.writer(f) for k, f in self.files.items()}
        self.w['trajectory'].writerow(['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        self.w['status'].writerow(['t', 'status'])
        self.w['offset'].writerow(['t', 'x', 'y', 'z'])
        self.map_frame = map_frame
        self.node.create_subscription(Odometry, odom_topic, self.odom, 100)
        self.node.create_subscription(String, '/localizer/localizer_status', self.status, 10)
        self.node.create_subscription(TFMessage, '/tf', self.tf, 100)
        self.thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.thread.start()

    @staticmethod
    def sec(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.w['trajectory'].writerow([f'{self.sec(m.header.stamp):.6f}', p.x, p.y, p.z, q.x, q.y, q.z, q.w])

    def status(self, m):
        self.w['status'].writerow([f'{time.time():.3f}', m.data])

    def tf(self, m):
        for t in m.transforms:
            if t.header.frame_id == self.map_frame:
                tr = t.transform.translation
                self.w['offset'].writerow([f'{self.sec(t.header.stamp):.6f}', tr.x, tr.y, tr.z])

    def close(self):
        self.rclpy.shutdown()
        self.thread.join(timeout=2)
        for f in self.files.values():
            f.close()


def load_csv(path, cols):
    rows = list(csv.reader(open(path)))[1:]
    if not rows:
        return np.zeros((0, cols))
    return np.array([[float(x) for x in r[:cols]] for r in rows])


def compute(out, with_localizer):
    tr = load_csv(out / 'trajectory.csv', 8)
    result = {'trajectory': metrics.trajectory_metrics(tr[:, 0], tr[:, 1:4]) if len(tr) else {'poses': 0}}
    if with_localizer:
        status = [(float(t), *metrics.parse_status(s)) for t, s in list(csv.reader(open(out / 'status.csv')))[1:]]
        off = load_csv(out / 'offset.csv', 4)
        result['localizer'] = metrics.localizer_metrics(status, off[:, 0] if len(off) else None,
                                                        off[:, 1:4] if len(off) else None)
    return result


class ResourceSampler:
    """Samples CPU (% of one core) and RSS of a process and its children."""

    def __init__(self, proc, period=0.5):
        import psutil
        self.psutil, self.root = psutil, psutil.Process(proc.pid)
        self.samples, self.period, self.stop_flag = [], period, threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def tree(self):
        """The process and its children, reusing Process objects: cpu_percent()
        measures since the previous call on the same object (a new one reads 0)."""
        try:
            procs = [self.root] + self.root.children(recursive=True)
        except self.psutil.Error:
            return []
        for p in procs:
            if p.pid not in self.known:
                self.known[p.pid] = p
                p.cpu_percent(None)             # prime; counts from the next sample
        return [self.known[p.pid] for p in procs]

    def loop(self):
        self.known = {}
        self.tree()
        while not self.stop_flag.wait(self.period):
            cpu = rss = 0.0
            for p in self.tree():
                try:
                    cpu += p.cpu_percent(None)
                    rss += p.memory_info().rss / 1e6
                except self.psutil.Error:
                    pass
            self.samples.append((cpu, rss))

    def close(self):
        self.stop_flag.set()
        self.thread.join(timeout=2)
        return metrics.resource_summary(self.samples)


def run(args):
    isolate(args.domain)
    import yaml
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.lio == 'upstream':
        if args.map or args.save_map:
            sys.exit('--map and --save-map need --lio fork (upstream FAST-LIO has no localizer or save service)')
        lio_cfg = args.lio_config or share('fast_lio', 'config', 'mid360.yaml')
        lio_cmd = ['ros2', 'run', 'fast_lio', 'fastlio_mapping', '--ros-args', '--params-file', lio_cfg]
        odom_topic = '/Odometry'
    else:
        lio_cfg = args.lio_config or share('fastlio2', 'config', 'lio.yaml')
        lio_cmd = ['ros2', 'run', 'fastlio2', 'lio_node', '--ros-args', '-r', '__ns:=/fastlio2',
                   '-p', f'config_path:={lio_cfg}']
        odom_topic = '/fastlio2/lio_odom'
    procs = [start(lio_cmd, out / 'lio.log')]
    map_frame = 'map'
    if args.map:
        cfg = yaml.safe_load(open(args.localizer_config or share('localizer', 'config', 'localizer.yaml')))
        cfg['map_path'] = str(Path(args.map).expanduser().resolve())
        cfg['pose_save_path'] = str(out / 'last_pose.yaml')   # never touch the real warm-start file
        (out / 'last_pose.yaml').unlink(missing_ok=True)          # start with a global search
        yaml.safe_dump(cfg, open(out / 'localizer.yaml', 'w'))
        map_frame = cfg.get('map_frame', 'map')
        procs.append(start(['ros2', 'run', 'localizer', 'localizer_node', '--ros-args', '-r', '__ns:=/localizer',
                            '-p', f'config_path:={out / "localizer.yaml"}'], out / 'localizer.log'))
    rec = Recorder(out, map_frame, odom_topic)
    resources, sampler = {'samples': 0}, None
    try:
        time.sleep(args.warmup)
        sampler = ResourceSampler(procs[0])     # during playback only
        subprocess.run(['ros2', 'bag', 'play', str(Path(args.bag).expanduser()), '--rate', str(args.rate)],
                       stdout=open(out / 'play.log', 'w'), stderr=subprocess.STDOUT, check=True)
        resources, sampler = sampler.close(), None
        time.sleep(3.0)
        if args.save_map:
            path = Path(args.save_map).expanduser().resolve()
            subprocess.run(['ros2', 'service', 'call', '/fastlio2/save_map', 'interface/srv/SaveMaps',
                            f"{{file_path: '{path}', save_patches: false}}"],
                           stdout=open(out / 'save_map.log', 'w'), stderr=subprocess.STDOUT, timeout=60)
    finally:
        if sampler:
            sampler.close()
        for p in procs:
            stop(p)
        rec.close()
    result = compute(out, bool(args.map))
    result['lio'] = {'stack': args.lio, 'config': str(lio_cfg), 'resources': resources}
    result['bag'] = str(Path(args.bag).expanduser().resolve())
    json.dump(result, open(out / 'metrics.json', 'w'), indent=2)
    print(json.dumps(result, indent=2))


def compare(args):
    a = load_csv(Path(args.a) / 'trajectory.csv', 8)
    b = load_csv(Path(args.b) / 'trajectory.csv', 8)
    print(json.dumps(metrics.compare_trajectories(a[:, 0], a[:, 1:4], a[:, 4:8],
                                                  b[:, 0], b[:, 1:4], b[:, 4:8]), indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='replay a recording and measure it')
    r.add_argument('--bag', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--lio', choices=('fork', 'upstream'), default='fork',
                   help="fork: this repo's lio_node; upstream: hku-mars FAST_LIO (as pinned by MappingUAV)")
    r.add_argument('--map', help='PCD map; also runs the localizer in global mode (fork only)')
    r.add_argument('--lio-config')
    r.add_argument('--localizer-config')
    r.add_argument('--save-map', help='save the LIO map to this PCD after playback')
    r.add_argument('--rate', type=float, default=1.0)
    r.add_argument('--domain', type=int, default=87)
    r.add_argument('--warmup', type=float, default=4.0, help='seconds between node start and playback')
    c = sub.add_parser('compare', help='compare the trajectories of two runs')
    c.add_argument('a')
    c.add_argument('b')
    args = ap.parse_args()
    run(args) if args.cmd == 'run' else compare(args)


if __name__ == '__main__':
    main()
