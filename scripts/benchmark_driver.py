"""
Driver-detection benchmark on the synthetic 3-D world (app/vehicle_monitoring/simulator.py).

    python scripts/benchmark_driver.py                       # all scenarios, forklift types, views (about 5 min)
    python scripts/benchmark_driver.py --quick               # a small subset (about 30 s)
    python scripts/benchmark_driver.py --seeds 3             # more random variation
    python scripts/benchmark_driver.py --no-contour          # vehicle model WITHOUT segmentation (boxes only)
    python scripts/benchmark_driver.py --set pairing.board_confirm_sec=1.0    # try a setting

It runs the REAL driver state machine with the settings of configs/vehicle_monitoring.yaml on simulated detections
(forklifts with / without roof / closed cabin, hidden body parts, noisy keypoints, bystanders, id changes) and compares
with the ground truth. Columns:
    correct   share of the time the right person is bound as the driver of a driven forklift
    wrong     another person is bound                                   <- the worst error
    missing   nobody / "unknown driver"
    falseDrv  a forklift that nobody drives has a "driver" (bystander)
    onboard   onboard / dismounted state right (while the right driver is bound)
    seatErr   error of the reported seat time
    alarmRec  "driver left the forklift" situations that raised an alarm (found / total), delay in seconds
    falseAl   alarms although the driver was at the forklift
Use it to check a change of the settings or of the code BEFORE trying it on the site.
It measures the logic under controlled difficulty - it is NOT the accuracy on your cameras.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from app.vehicle_monitoring import simulator as S  # noqa: E402

HEADER = (f"{'scenario':20s} {'type':6s} {'correct':>8s} {'wrong':>7s} {'missing':>8s} {'falseDrv':>9s} "
          f"{'onboard':>8s} {'seatErr':>8s} {'alarmRec':>10s} {'delay':>6s} {'falseAl':>8s}")


def row(name: str, kind: str, sc: S.Scores) -> str:
    m = sc.summary()
    return (f"{name:20s} {kind:6s} {m['driver_correct'] * 100:7.1f}% {m['driver_wrong'] * 100:6.1f}% "
            f"{m['driver_missing'] * 100:7.1f}% {m['false_driver'] * 100:8.1f}% {m['onboard_state'] * 100:7.1f}% "
            f"{m['seat_time_error'] * 100:7.1f}% {int(round(m['alarm_recall'] * m['alarms_true'])):4d}/"
            f"{int(m['alarms_true']):<4d} {m['alarm_delay_sec']:6.1f} {int(m['false_alarms']):8d}")


def apply_overrides(cfg: dict, items) -> dict:
    for item in items or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Driver-detection benchmark (synthetic ground truth)")
    ap.add_argument("--config", default="configs/vehicle_monitoring.yaml")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--scenario", nargs="*", default=None, choices=list(S.SCENARIOS))
    ap.add_argument("--kind", nargs="*", default=None, choices=list(S.KINDS))
    ap.add_argument("--no-contour", action="store_true", help="vehicle detections without a segmentation outline")
    ap.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE", help="override a setting (repeatable)")
    args = ap.parse_args(argv)

    cfg = apply_overrides(S.load_fsm_config(args.config), args.set)
    kw = dict(scenarios=args.scenario, kinds=tuple(args.kind) if args.kind else S.KINDS, seeds=tuple(range(args.seeds)),
              config=cfg, use_contour=not args.no_contour)
    if args.quick:
        kw.update(scenarios=args.scenario or ["board_drive_alarm", "bystander_behind", "driver_change"],
                  headings=(0.0, 270.0), cameras=["oblique", "steep"])
    t0 = time.time()
    print(HEADER)
    res = S.run_benchmark(progress=None, **kw)
    per_kind = {k: S.Scores() for k in kw["kinds"]}
    total = S.Scores()
    for (sc, kind), s in res.items():
        print(row(sc, kind, s))
        per_kind[kind].add(s)
        total.add(s)
    print("-" * len(HEADER))
    for k, s in per_kind.items():
        print(row("ALL", k, s))
    print(row("ALL", "all", total))
    print(f"{int(total.runs)} runs, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
