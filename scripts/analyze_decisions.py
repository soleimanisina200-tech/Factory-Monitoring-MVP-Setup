"""
Summarise the identity decision log (logs/identity_decisions.jsonl) into a short text report.

The report shows how often each identity decision happened, the similarity values behind them, and a
list of "moments to review" (timestamps where an ID error is most likely). Send the report text back
for threshold tuning instead of the raw log.

Usage:
    python scripts/analyze_decisions.py
    python scripts/analyze_decisions.py --log logs/identity_decisions.jsonl --out logs/identity_report.txt
    python scripts/analyze_decisions.py --json logs/identity_report.json

Timestamps: for video FILES the log uses video time, so mm:ss matches the position in the video.
For live streams the log uses a monotonic clock; use --relative to print time since the first event.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_THRESHOLDS = {
    "reentry_sim_thresh": 0.82,
    "stitch_sim_thresh": 0.55,
    "spatial_recovery_sim_thresh": 0.48,
    "cos_sim_thresh": 0.60,
    "reentry_margin": 0.08,
}


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------
def load_events(paths: Iterable[str]) -> List[Dict[str, Any]]:
    """Read JSON lines from the given files (missing files are skipped, bad lines are counted)."""
    events: List[Dict[str, Any]] = []
    bad = 0
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if isinstance(ev, dict) and "event" in ev and "ts" in ev:
                    events.append(ev)
                else:
                    bad += 1
    events.sort(key=lambda e: float(e["ts"]))
    if bad:
        events.append({"event": "_BAD_LINES", "ts": events[-1]["ts"] if events else 0.0, "count": bad})
    return events


def load_thresholds(settings_path: Optional[str]) -> Dict[str, float]:
    th = dict(DEFAULT_THRESHOLDS)
    if not settings_path or not Path(settings_path).exists():
        return th
    try:
        import yaml

        with open(settings_path, "r", encoding="utf-8") as f:
            reid = (yaml.safe_load(f) or {}).get("reid", {}) or {}
        for k in th:
            if k in reid:
                th[k] = float(reid[k])
    except Exception:
        pass
    return th


# ---------------------------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------------------------
def _stats(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    v = sorted(values)

    def pct(q: float) -> float:
        return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]

    return {
        "n": len(v),
        "min": round(v[0], 3),
        "p10": round(pct(0.10), 3),
        "median": round(statistics.median(v), 3),
        "p90": round(pct(0.90), 3),
        "max": round(v[-1], 3),
    }


def _histogram(values: List[float], lo: float = 0.0, hi: float = 1.0, bins: int = 10) -> List[int]:
    counts = [0] * bins
    for x in values:
        idx = int((min(max(x, lo), hi - 1e-9) - lo) / (hi - lo) * bins)
        counts[idx] += 1
    return counts


def summarize(
    events: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
    near_margin: float = 0.05,
    quick_return_sec: float = 3.0,
) -> Dict[str, Any]:
    th = dict(DEFAULT_THRESHOLDS)
    th.update(thresholds or {})
    real = [e for e in events if not e["event"].startswith("_")]
    bad_lines = sum(e.get("count", 0) for e in events if e["event"] == "_BAD_LINES")

    counts = Counter(e["event"] for e in real)
    t0 = float(real[0]["ts"]) if real else 0.0
    t1 = float(real[-1]["ts"]) if real else 0.0
    minutes = max((t1 - t0) / 60.0, 1e-9)

    matched_by_stage: Dict[str, List[float]] = defaultdict(list)
    reentries: List[Dict[str, Any]] = []
    new_id_best: List[float] = []
    ambiguous: List[Dict[str, Any]] = []
    merges: List[Dict[str, Any]] = []
    broken: List[Dict[str, Any]] = []
    exits_by: Counter = Counter()
    resurrect_gaps: List[float] = []
    review: List[Dict[str, Any]] = []
    last_exit_ts: Optional[float] = None
    new_id_ts: Dict[Any, float] = {}

    for e in real:
        ev, ts = e["event"], float(e["ts"])
        if ev == "MATCHED":
            sim = float(e.get("sim", 0.0))
            stage = str(e.get("stage", "UNKNOWN"))
            matched_by_stage[stage].append(sim)
            if e.get("status_before") == "EXITED":
                reentries.append({"ts": ts, "gid": e.get("gid"), "sim": sim, "dt_lost": e.get("dt_lost")})
                if sim < th["reentry_sim_thresh"] + near_margin:
                    review.append({"ts": ts, "kind": "REENTRY_NEAR_THRESHOLD", "gid": e.get("gid"),
                                   "detail": f"sim={sim:.3f} (threshold {th['reentry_sim_thresh']:.2f}); "
                                             "check it is really the same person"})
            elif stage == "STITCH" and sim < th["stitch_sim_thresh"] + near_margin:
                review.append({"ts": ts, "kind": "STITCH_NEAR_THRESHOLD", "gid": e.get("gid"),
                               "detail": f"sim={sim:.3f} (threshold {th['stitch_sim_thresh']:.2f})"})
        elif ev == "NEW_ID":
            best = float(e.get("best_existing_sim", 0.0))
            new_id_best.append(best)
            new_id_ts[e.get("gid")] = ts
            if best >= th["reentry_sim_thresh"] - near_margin:
                review.append({"ts": ts, "kind": "NEW_ID_BUT_LOOKS_KNOWN", "gid": e.get("gid"),
                               "detail": f"best existing sim={best:.3f}: same person split into a new ID, "
                                         "or a lookalike correctly kept apart?"})
            elif last_exit_ts is not None and ts - last_exit_ts <= quick_return_sec and best >= th["cos_sim_thresh"]:
                review.append({"ts": ts, "kind": "NEW_ID_RIGHT_AFTER_EXIT", "gid": e.get("gid"),
                               "detail": f"{ts - last_exit_ts:.1f}s after an exit, best existing sim={best:.3f}"})
        elif ev in ("MATCH_REJECTED_AMBIGUOUS", "MERGE_REJECTED_AMBIGUOUS"):
            sim = float(e.get("sim", e.get("gallery_sim", 0.0)))
            runner = float(e.get("runner_up", 0.0))
            ambiguous.append({"ts": ts, "event": ev, "gid": e.get("gid", e.get("gid_old")), "sim": sim,
                              "runner_up": runner})
            review.append({"ts": ts, "kind": "AMBIGUOUS_LOOKALIKES", "gid": e.get("gid", e.get("gid_old")),
                           "detail": f"sim={sim:.3f} vs runner-up {runner:.3f}: two workers look almost the same"})
        elif ev == "MERGED":
            gid_new = e.get("gid_new")
            created = new_id_ts.get(gid_new)
            merges.append({"ts": ts, "gid_new": gid_new, "gid_old": e.get("gid_old"),
                           "sim": float(e.get("gallery_sim", 0.0)),
                           "id_lifetime_sec": None if created is None else round(ts - created, 2)})
        elif ev == "BINDING_BROKEN":
            broken.append({"ts": ts, "gid": e.get("gid"), "sim": e.get("sim"), "streak_sec": e.get("streak_sec")})
            review.append({"ts": ts, "kind": "BINDING_BROKEN", "gid": e.get("gid"),
                           "detail": f"tracker probably jumped to another person (sim={e.get('sim')})"})
        elif ev == "SWAP_FIXED":
            review.append({"ts": ts, "kind": "SWAP_FIXED", "gid": e.get("gid_a"),
                           "detail": f"IDs {e.get('gid_a')} and {e.get('gid_b')} were swapped and corrected"})
        elif ev == "EXITED":
            exits_by[f"{e.get('reason', '?')}:{e.get('edge') or '-'}"] += 1
            last_exit_ts = ts
        elif ev == "TRACK_RESURRECTED":
            resurrect_gaps.append(float(e.get("gap_sec", 0.0)))

    all_matched = [s for v in matched_by_stage.values() for s in v]
    return {
        "time_span": {"start_ts": t0, "end_ts": t1, "minutes": round((t1 - t0) / 60.0, 2)},
        "bad_lines": bad_lines,
        "event_counts": dict(counts.most_common()),
        "rates_per_minute": {
            "new_ids": round(counts.get("NEW_ID", 0) / minutes, 2),
            "exits": round(counts.get("EXITED", 0) / minutes, 2),
            "merges": round(counts.get("MERGED", 0) / minutes, 2),
            "broken_bindings": round(counts.get("BINDING_BROKEN", 0) / minutes, 2),
        },
        "thresholds": th,
        "matched_sim_by_stage": {k: _stats(v) for k, v in sorted(matched_by_stage.items())},
        "matched_sim_histogram": _histogram(all_matched),
        "new_id_best_existing_sim": _stats(new_id_best),
        "new_id_best_existing_histogram": _histogram(new_id_best),
        "reentries": reentries,
        "ambiguous_rejections": ambiguous,
        "merges": merges,
        "broken_bindings": broken,
        "exits_by_reason_edge": dict(exits_by),
        "track_resurrection_gap_sec": _stats(resurrect_gaps),
        "review_moments": sorted(review, key=lambda r: r["ts"]),
    }


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------
def _fmt_ts(ts: float, t0: float, relative: bool) -> str:
    t = ts - t0 if relative else ts
    sign = "-" if t < 0 else ""
    t = abs(t)
    return f"{sign}{int(t // 60):02d}:{t % 60:05.2f}"


def _fmt_stats(s: Optional[Dict[str, float]]) -> str:
    if not s:
        return "-"
    return f"n={s['n']}  min={s['min']}  p10={s['p10']}  median={s['median']}  p90={s['p90']}  max={s['max']}"


def _bar_rows(hist: List[int]) -> List[str]:
    peak = max(hist) if hist and max(hist) > 0 else 1
    rows = []
    for i, c in enumerate(hist):
        lo, hi = i / len(hist), (i + 1) / len(hist)
        rows.append(f"    {lo:.1f}-{hi:.1f} | {'#' * int(round(30 * c / peak)):<30} {c}")
    return rows


def render_text(summary: Dict[str, Any], relative: bool = False, top: int = 40) -> str:
    t0 = summary["time_span"]["start_ts"]
    th = summary["thresholds"]
    out: List[str] = []
    out.append("IDENTITY DECISION REPORT")
    out.append("=" * 72)
    span = summary["time_span"]
    out.append(f"Covered: {_fmt_ts(span['start_ts'], t0, relative)} -> {_fmt_ts(span['end_ts'], t0, relative)}"
               f"  ({span['minutes']} min)" + ("   [relative time]" if relative else "   [video time]"))
    if summary["bad_lines"]:
        out.append(f"WARNING: {summary['bad_lines']} unreadable lines skipped")
    out.append("Thresholds: " + ", ".join(f"{k}={v}" for k, v in th.items()))

    out.append("\n1) Event counts")
    for k, v in summary["event_counts"].items():
        out.append(f"    {k:<26} {v}")
    r = summary["rates_per_minute"]
    out.append(f"    per minute: new IDs {r['new_ids']}, exits {r['exits']}, merges {r['merges']}, "
               f"broken bindings {r['broken_bindings']}")

    out.append("\n2) Similarity of ACCEPTED matches, by stage")
    for stage, s in summary["matched_sim_by_stage"].items():
        out.append(f"    {stage:<16} {_fmt_stats(s)}")
    out.extend(_bar_rows(summary["matched_sim_histogram"]))

    out.append("\n3) New IDs: best similarity to an EXISTING identity at creation")
    out.append("    (high values = the system saw someone similar but still created a new ID)")
    out.append("    " + _fmt_stats(summary["new_id_best_existing_sim"]))
    out.extend(_bar_rows(summary["new_id_best_existing_histogram"]))

    out.append("\n4) Re-entries of workers who had LEFT the camera")
    if not summary["reentries"]:
        out.append("    none")
    for x in summary["reentries"][:top]:
        dt = x.get("dt_lost")
        out.append(f"    {_fmt_ts(x['ts'], t0, relative)}  ID {x['gid']}  sim={x['sim']:.3f}"
                   + (f"  away {float(dt):.1f}s" if dt is not None else ""))

    out.append("\n5) Merges (a temporary ID folded back into an older one)")
    if not summary["merges"]:
        out.append("    none")
    for m in summary["merges"][:top]:
        life = m["id_lifetime_sec"]
        out.append(f"    {_fmt_ts(m['ts'], t0, relative)}  ID {m['gid_new']} -> {m['gid_old']}  "
                   f"gallery_sim={m['sim']:.3f}" + (f"  (temporary ID lived {life}s)" if life is not None else ""))

    out.append("\n6) Ambiguous lookalikes (match refused because two workers look alike)")
    if not summary["ambiguous_rejections"]:
        out.append("    none")
    for a in summary["ambiguous_rejections"][:top]:
        out.append(f"    {_fmt_ts(a['ts'], t0, relative)}  ID {a['gid']}  sim={a['sim']:.3f}  "
                   f"runner-up={a['runner_up']:.3f}  gap={a['sim'] - a['runner_up']:.3f}")

    out.append("\n7) Exits by reason:edge")
    for k, v in sorted(summary["exits_by_reason_edge"].items()):
        out.append(f"    {k:<20} {v}")
    out.append("    track resurrections (gap s): " + _fmt_stats(summary["track_resurrection_gap_sec"]))

    moments = summary["review_moments"]
    out.append(f"\n8) MOMENTS TO REVIEW IN THE VIDEO ({len(moments)} total, first {min(top, len(moments))} shown)")
    if not moments:
        out.append("    none")
    for mo in moments[:top]:
        out.append(f"    {_fmt_ts(mo['ts'], t0, relative)}  [{mo['kind']}]  ID {mo['gid']}: {mo['detail']}")

    out.append("\nNext: open the video at the times in section 8 and note which ones were real ID errors.")
    return "\n".join(out) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Summarise the identity decision log")
    ap.add_argument("--log", default="logs/identity_decisions.jsonl", help="decision log (JSON lines)")
    ap.add_argument("--include-rotated", action="store_true", help="also read the rotated <log>.1 file")
    ap.add_argument("--settings", default="configs/settings.yaml", help="read thresholds from this settings file")
    ap.add_argument("--relative", action="store_true", help="print time since the first event (live streams)")
    ap.add_argument("--top", type=int, default=40, help="max rows per list")
    ap.add_argument("--out", default=None, help="also write the text report to this file")
    ap.add_argument("--json", default=None, help="write the full summary as JSON to this file")
    args = ap.parse_args(argv)

    paths = ([args.log + ".1"] if args.include_rotated else []) + [args.log]
    events = load_events(paths)
    if not [e for e in events if not e["event"].startswith("_")]:
        print(f"No decision events found in {args.log}. Is reid.decision_log_path set in configs/settings.yaml "
              "and did main.py run?")
        return 1

    summary = summarize(events, load_thresholds(args.settings))
    report = render_text(summary, relative=args.relative, top=args.top)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
