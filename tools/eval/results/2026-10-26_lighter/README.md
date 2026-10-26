# Lighter `lio_node`: same-day replay of the 2026-10-23 X500 recordings

Goal: bring the fork's cost down to roughly upstream's before swapping it onto the drone (CPU within
about 1.2 times upstream), without losing accuracy (settled gap no more than about 1 cm worse on any
recording). Replayed with `tools/eval/eval.py` on the Jetson Orin Nano, all variants on the same day:
replays are deterministic within a session but drift by about 3 mm RMS (19 mm max) between days, so
only same-day runs are compared.

`summary.txt` is the table; `runs/<variant>_<bag>/` holds each run's `metrics.json` and
`trajectory.csv`. `run_all.sh` and `summarize.py` produced them; `lio_x500_lean.yaml` is the matched
config with all four optional outputs off.

## Variants

| Variant | Commit | Change |
|---|---|---|
| upstream | hku-mars `a4743b09` | MappingUAV's drone config |
| base | `2a403fb` | per-scan buffer sizing, planes fitted on the stack, optional outputs off |
| r1 | `3d4703f` | base + data race on the point selection flags fixed |
| l4 | `b96314c` | r1 + nearest neighbours searched only after a small step, as upstream does |

## Where the CPU went

Profiled with perf: the time is in kd-tree nearest-neighbour search. The fork searched about twice
as often as upstream, because it re-searched on every filter iteration and its tighter stop rule ran
more iterations. Memory was mostly the always-on accumulated map (416 MB to 158 MB once optional).

## Results

| Bag | upstream CPU (p95) / gap | base | **l4** | memory upstream / l4 |
|---|---|---|---|---|
| still5 | 29 % (34) / 3.7 cm | 46 % (59) / 2.8 cm | **23 % (26) / 3.1 cm** | 172 / 157 MB |
| loop5 | 29 % (32) / 13.0 cm | 54 % (70) / 10.0 cm | **23 % (26) / 9.5 cm** | 176 / 163 MB |
| height | 26 % (30) / 5.7 cm | 43 % (57) / 6.3 cm | **21 % (25) / 5.8 cm** | 171 / 160 MB |
| yaw | 29 % (32) / 9.5 cm | 53 % (72) / 8.9 cm | **23 % (26) / 10.0 cm** | 176 / 158 MB |
| hallway | 62 % (70) / 4.3 cm | 111 % (145) / 3.8 cm | **44 % (51) / 3.6 cm** | 178 / 166 MB |

CPU is % of one core; gap is the settled return gap (from the pose 8 s in, see the A3 README).

- **Cost:** l4 runs at about 0.75 times upstream's CPU (was 1.8 times), with lower peaks, and below
  upstream's memory on every recording.
- **Accuracy:** l4's settled gap changes by -0.5 to +1.1 cm against base: better on three, worse on
  `still5` (+0.3) and `yaw` (+1.1, just over the 1 cm bar but within 0.5 cm of upstream).
- **r1** is bit-identical to base on all five: the race did not change results on these runs.
- **l4's path** differs from base by up to 7.5 cm mid-run (RMS 3 to 23 mm). Against upstream, base
  and l4 differ by the same amount (2 to 9 cm RMS), so l4 does not move away from upstream.
- No warnings or errors in any run. Upstream's stop rule (a further option) was not needed.
