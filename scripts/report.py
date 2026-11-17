"""
Per-worker report from the event database (data/factory_events.db).

    python scripts/report.py                                    # today
    python scripts/report.py --date 2026-11-17 --out reports/2026-11-17
    python scripts/report.py --from "2026-11-17 06:00" --to "2026-11-17 14:00" --camera cam1

Writes (UTF-8 CSV, opens in Excel):
    workers.csv            one row per worker: zone time per zone, crossings per line/direction, forklift seat time,
                           alarms
    zone_visits.csv        every visit: worker, zone, enter, exit, seconds
    crossings.csv          every line/region passage with direction
    forklift_sessions.csv  every driver session: forklift, driver, start, end, seat seconds, end reason
    alarms.csv             every warning/alarm
    summary.txt            human-readable summary
Temporary IDs that were later merged into another ID are attributed to the surviving worker (identity_aliases).
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _fmt(epoch: Optional[float]) -> str:
    return "" if epoch is None else datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")


def _hms(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600:d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


class AliasResolver:
    """Local worker IDs are per camera: (camera, temporary id) -> (camera, surviving id)."""

    def __init__(self, rows: List[Tuple[str, int, int]]):
        self.map = {(str(c), int(o)): int(n) for c, o, n in rows}

    def __call__(self, camera: str, wid: Optional[int]) -> Optional[int]:
        seen = set()
        while wid is not None and (camera, wid) in self.map and wid not in seen:
            seen.add(wid)
            wid = self.map[(camera, wid)]
        return wid


def load(db: str, t0: float, t1: float, camera: Optional[str]):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cam_sql, cam_args = ("AND camera_id = ?", [camera]) if camera else ("", [])
    events = [dict(r) for r in conn.execute(
        f"SELECT * FROM events WHERE wall_epoch >= ? AND wall_epoch < ? {cam_sql} ORDER BY wall_epoch, id",
        [t0, t1] + cam_args)]
    aliases = [(r["camera_id"], r["old_id"], r["new_id"]) for r in conn.execute(
        f"SELECT camera_id, old_id, new_id FROM identity_aliases WHERE 1=1 {cam_sql}", cam_args)]
    labels = {(r["camera_id"], r["worker_id"]): r["label"] for r in conn.execute(
        f"SELECT camera_id, worker_id, label FROM worker_labels WHERE 1=1 {cam_sql}", cam_args)}
    conn.close()
    for e in events:
        try:
            e["details"] = json.loads(e.get("details") or "{}")
        except json.JSONDecodeError:
            e["details"] = {}
    return events, AliasResolver(aliases), labels


def load_persons(registry: Optional[str]) -> Dict[Tuple[str, int], int]:
    """(camera, local worker id) -> cross-camera person id (empty when there is no registry)."""
    if not registry or not Path(registry).exists():
        return {}
    conn = sqlite3.connect(f"file:{registry}?mode=ro", uri=True)
    merged = {int(p): int(m) for p, m in conn.execute(
        "SELECT person_id, merged_into FROM persons WHERE merged_into IS NOT NULL")}
    out = {}
    for cam, lid, pid in conn.execute(
            "SELECT camera_id, local_id, person_id FROM tracklets WHERE person_id IS NOT NULL"):
        seen, p = set(), int(pid)
        while p in merged and p not in seen:
            seen.add(p)
            p = merged[p]
        out[(str(cam), int(lid))] = p
    conn.close()
    return out


def build(events: List[Dict[str, Any]], resolve: AliasResolver, labels: Dict[Tuple[str, int], str],
          persons: Optional[Dict[Tuple[str, int], int]] = None) -> Dict[str, Any]:
    persons = persons or {}
    multi = len({e["camera_id"] for e in events}) > 1
    names: Dict[str, str] = {}

    def who(e: Dict[str, Any]) -> str:
        cam = str(e["camera_id"])
        wid = resolve(cam, e.get("worker_id"))
        if wid is None:
            return f"unidentified({cam}:T{e['track_id']})" if e.get("track_id") is not None else "unidentified"
        pid = persons.get((cam, wid))
        key = f"P{pid}" if pid is not None else (f"{cam}:{wid}" if multi else str(wid))
        lab = labels.get((cam, wid)) or e.get("worker_label")
        if lab:
            names.setdefault(key, lab)
        return key

    def label(w: str) -> str:
        return names.get(w, "")

    workers: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "first": None, "last": None, "zone_sec": defaultdict(float), "zone_visits": defaultdict(int),
        "cross": defaultdict(int), "seat_sec": 0.0, "sessions": 0, "alarms": 0, "warnings": 0,
    })
    visits, crossings, sessions, alarms = [], [], [], []
    for e in events:   # pre-pass: learn every worker's name before any row is written
        if e.get("worker_id") is not None:
            who(e)
    for e in events:
        if e["type"] == "PERSON_LINKED":
            continue   # bookkeeping of the cross-camera registry, not worker activity
        w = who(e) if (e.get("worker_id") is not None or e.get("track_id") is not None) else None
        cam = e["camera_id"]
        t = float(e["wall_epoch"])
        if w is not None:
            rec = workers[w]
            rec["first"] = t if rec["first"] is None else min(rec["first"], t)
            rec["last"] = t if rec["last"] is None else max(rec["last"], t)
        etype, d = e["type"], e["details"]
        if etype == "ZONE_EXIT" and w is not None:
            dwell = float(e.get("value") or 0.0)
            start = t - dwell
            workers[w]["zone_sec"][e["subject"]] += dwell
            workers[w]["zone_visits"][e["subject"]] += 1
            workers[w]["first"] = min(workers[w]["first"], start)
            visits.append([w, label(w), cam, e["subject"], _fmt(start), _fmt(t), round(dwell, 1),
                           d.get("reason", "")])
        elif etype in ("LINE_CROSS", "REGION_PASS") and w is not None:
            workers[w]["cross"][f"{e['subject']}:{e['direction']}"] += 1
            crossings.append([w, label(w), cam, _fmt(t), "line" if etype == "LINE_CROSS" else "region",
                              e["subject"], e["direction"]])
        elif etype == "DRIVER_SESSION_END":
            onboard = float(d.get("onboard_sec", e.get("value") or 0.0))
            session = float(d.get("session_sec") or onboard)
            start = t - session
            drv = w or "unknown"
            if w is not None:
                workers[w]["seat_sec"] += onboard
                workers[w]["sessions"] += 1
            sessions.append([cam, e.get("vehicle_id"), drv, label(drv), _fmt(start), _fmt(t), round(onboard, 1),
                             round(session, 1), d.get("reason", "")])
        if e.get("severity") in ("alarm", "warning"):
            if w is not None:
                workers[w]["alarms" if e["severity"] == "alarm" else "warnings"] += 1
            alarms.append([_fmt(t), cam, e["severity"], etype, w or "", label(w) if w else "",
                           e.get("vehicle_id") or "", e.get("subject") or "",
                           "" if e.get("value") is None else round(float(e["value"]), 2),
                           e.get("snapshot_path") or ""])
    return {"workers": workers, "visits": visits, "crossings": crossings, "sessions": sessions, "alarms": alarms,
            "label": label}


def write(report: Dict[str, Any], out: Path, period: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    workers, label = report["workers"], report["label"]
    zones = sorted({z for w in workers.values() for z in w["zone_sec"]})
    cross = sorted({c for w in workers.values() for c in w["cross"]})

    def wcsv(name: str, header: List[str], rows: List[List[Any]]) -> None:
        with open(out / name, "w", newline="", encoding="utf-8-sig") as f:
            cw = csv.writer(f)
            cw.writerow(header)
            cw.writerows(rows)

    def sort_key(k: str):
        return (0, int(k)) if k.isdigit() else (1, k)

    rows = []
    for w in sorted(workers, key=sort_key):
        r = workers[w]
        rows.append([w, label(w), _fmt(r["first"]), _fmt(r["last"])]
                    + [round(r["zone_sec"].get(z, 0.0), 1) for z in zones]
                    + [r["cross"].get(c, 0) for c in cross]
                    + [r["sessions"], round(r["seat_sec"], 1), r["warnings"], r["alarms"]])
    wcsv("workers.csv", ["worker_id", "label", "first_event", "last_event"] + [f"zone_sec:{z}" for z in zones]
         + [f"count:{c}" for c in cross] + ["forklift_sessions", "forklift_seat_sec", "warnings", "alarms"], rows)
    wcsv("zone_visits.csv", ["worker_id", "label", "camera", "zone", "enter", "exit", "seconds", "end_reason"],
         report["visits"])
    wcsv("crossings.csv", ["worker_id", "label", "camera", "time", "kind", "name", "direction"],
         report["crossings"])
    wcsv("forklift_sessions.csv", ["camera", "forklift", "driver", "label", "start", "end", "seat_sec",
                                   "session_sec", "end_reason"], report["sessions"])
    wcsv("alarms.csv", ["time", "camera", "severity", "type", "worker_id", "label", "forklift", "subject", "value",
                        "snapshot"], report["alarms"])

    lines = [f"FACTORY MONITORING REPORT  {period}", "=" * 72, f"workers with events: {len(workers)}", ""]
    for w in sorted(workers, key=sort_key):
        r = workers[w]
        name = f"{w} ({label(w)})" if label(w) else w
        lines.append(f"Worker {name}: {_fmt(r['first'])} -> {_fmt(r['last'])}")
        for z in zones:
            if r["zone_visits"].get(z):
                lines.append(f"    zone {z:<20} {r['zone_visits'][z]:>3} visits  {_hms(r['zone_sec'][z])}")
        for c in cross:
            if r["cross"].get(c):
                lines.append(f"    passed {c:<24} {r['cross'][c]:>3}x")
        if r["sessions"]:
            lines.append(f"    forklift: {r['sessions']} sessions, seat time {_hms(r['seat_sec'])}")
        if r["alarms"] or r["warnings"]:
            lines.append(f"    alarms {r['alarms']}, warnings {r['warnings']}")
    lines += ["", f"alarms/warnings: {len(report['alarms'])}  (alarms.csv)",
              f"forklift sessions: {len(report['sessions'])}  (forklift_sessions.csv)"]
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text, encoding="utf-8")
    return text


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Per-worker report from the event database")
    ap.add_argument("--db", default="data/factory_events.db")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--from", dest="t_from", default=None, help="'YYYY-MM-DD HH:MM' (overrides --date)")
    ap.add_argument("--to", dest="t_to", default=None)
    ap.add_argument("--camera", default=None, help="only this camera (default: all cameras)")
    ap.add_argument("--registry", default="data/cross_camera.db",
                    help="cross-camera registry: workers linked across cameras are reported as one person (P#)")
    ap.add_argument("--out", default=None, help="output folder (default reports/<date>)")
    args = ap.parse_args(argv)
    if not Path(args.db).exists():
        print(f"Database not found: {args.db}")
        return 1
    if args.t_from:
        t0 = datetime.fromisoformat(args.t_from)
        t1 = datetime.fromisoformat(args.t_to) if args.t_to else t0 + timedelta(days=1)
    else:
        day = datetime.fromisoformat(args.date) if args.date else datetime.now().replace(hour=0, minute=0, second=0,
                                                                                        microsecond=0)
        t0, t1 = day, day + timedelta(days=1)
    events, resolve, labels = load(args.db, t0.timestamp(), t1.timestamp(), args.camera)
    out = Path(args.out or f"reports/{t0.strftime('%Y-%m-%d')}")
    text = write(build(events, resolve, labels, load_persons(args.registry)), out,
                 f"{t0:%Y-%m-%d %H:%M} -> {t1:%Y-%m-%d %H:%M}")
    sys.stdout.write(text)
    print(f"\nCSV files written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
