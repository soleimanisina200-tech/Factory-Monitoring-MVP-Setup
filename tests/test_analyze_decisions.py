"""The decision-log report: parsing, statistics and the 'moments to review' list."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from test_identity_reentry import Scenario, W, unit, with_similarity

_spec = importlib.util.spec_from_file_location(
    "analyze_decisions", Path(__file__).resolve().parents[1] / "scripts" / "analyze_decisions.py"
)
ad = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ad)


def write_log(path, events):
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        f.write("not json\n")


def test_report_flags_the_right_moments(tmp_path):
    log = tmp_path / "d.jsonl"
    write_log(log, [
        {"event": "NEW_ID", "ts": 10.0, "tid": 1, "gid": 1, "best_existing_sim": 0.0, "seen_frames": 4},
        {"event": "EXITED", "ts": 20.0, "gid": 1, "reason": "BORDER", "edge": "right", "absent_sec": 0.9},
        {"event": "NEW_ID", "ts": 21.0, "tid": 5, "gid": 2, "best_existing_sim": 0.80, "seen_frames": 4},
        {"event": "MATCHED", "ts": 30.0, "tid": 7, "gid": 1, "stage": "REENTRY_STRICT", "sim": 0.84,
         "dt_lost": 10.0, "status_before": "EXITED", "near_border": True},
        {"event": "MATCH_REJECTED_AMBIGUOUS", "ts": 40.0, "tid": 8, "gid": 1, "stage": "REENTRY_STRICT",
         "sim": 0.90, "runner_up": 0.88, "margin_needed": 0.08},
        {"event": "BINDING_BROKEN", "ts": 50.0, "tid": 9, "gid": 3, "sim": 0.3, "streak_sec": 1.2},
        {"event": "MERGED", "ts": 52.0, "tid": 9, "gid_new": 2, "gid_old": 1, "gallery_sim": 0.93},
    ])
    events = ad.load_events([str(log)])
    s = ad.summarize(events)

    assert s["bad_lines"] == 1
    assert s["event_counts"]["NEW_ID"] == 2
    assert s["reentries"][0]["gid"] == 1
    assert s["merges"][0]["id_lifetime_sec"] == 31.0
    kinds = [m["kind"] for m in s["review_moments"]]
    assert kinds == [
        "NEW_ID_BUT_LOOKS_KNOWN",     # 0.80 >= 0.82 - 0.05
        "REENTRY_NEAR_THRESHOLD",     # 0.84 < 0.82 + 0.05
        "AMBIGUOUS_LOOKALIKES",
        "BINDING_BROKEN",
    ]

    text = ad.render_text(s)
    assert "00:21.00" in text and "MOMENTS TO REVIEW" in text


def test_report_on_a_real_log_written_by_the_identity_manager(tmp_path):
    log = tmp_path / "decisions.jsonl"
    sc = Scenario(decision_log_path=str(log))
    sc.walk_out_right(1, unit(0))
    sc.idle(1.0)
    for k in range(10):
        sc.step(5, W - 70.0 - 5 * k, with_similarity(unit(0), 1, 0.86))
    sc.mgr.close()

    out_txt, out_json = tmp_path / "r.txt", tmp_path / "r.json"
    rc = ad.main(["--log", str(log), "--settings", "missing.yaml", "--out", str(out_txt), "--json", str(out_json)])
    assert rc == 0
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["event_counts"]["EXITED"] == 1
    assert summary["exits_by_reason_edge"] == {"BORDER:right": 1}
    # the lookalike got a new ID although it resembled the departed worker -> listed for review
    assert any(m["kind"] == "NEW_ID_BUT_LOOKS_KNOWN" for m in summary["review_moments"])
    assert "IDENTITY DECISION REPORT" in out_txt.read_text(encoding="utf-8")


def test_empty_or_missing_log_is_reported_clearly(tmp_path, capsys):
    assert ad.main(["--log", str(tmp_path / "nope.jsonl")]) == 1
    assert "No decision events" in capsys.readouterr().out


def test_histogram_and_stats_helpers():
    assert ad._histogram([0.05, 0.95, 1.0, 0.5]) == [1, 0, 0, 0, 0, 1, 0, 0, 0, 2]
    st = ad._stats(list(np.linspace(0, 1, 11)))
    assert st["min"] == 0.0 and st["max"] == 1.0 and st["median"] == 0.5
