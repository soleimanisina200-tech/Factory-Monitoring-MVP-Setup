"""
Vest markers (chest / back), marker size for a camera, and: a marker read on a track that was given an existing
identity after a hide (truck) belongs to that track, never to the hidden worker's earlier history.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.identity.markers import HelmetMarkerReader, WorkerRegistry, required_marker_cm
from test_identity_occluders import TRUCK_X2, Site, sim_to, unit

ROOT = Path(__file__).resolve().parent.parent


def _aruco_available():
    return hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector")


def _person_with_vest_marker(marker_id=3, marker_px=64, y0=150):
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id,
                                           marker_px)
    marker = cv2.copyMakeBorder(marker, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    frame = np.full((600, 400, 3), 110, np.uint8)
    s = marker.shape[0]
    frame[y0:y0 + s, 200 - s // 2:200 - s // 2 + s] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)   # on the chest
    box = np.array([150, 30, 250, 430], np.float32)
    kxy = np.zeros((17, 2), np.float32)
    kc = np.zeros(17, np.float32)
    for i, (x, y) in {5: (170, 110), 6: (230, 110), 11: (180, 230), 12: (220, 230)}.items():
        kxy[i], kc[i] = (x, y), 0.9                 # seen from behind: shoulders and hips, no face
    return frame, box, kxy, kc


@pytest.mark.skipif(not _aruco_available(), reason="OpenCV without ArUco")
def test_marker_on_the_vest_is_read_from_the_torso_region():
    reg = WorkerRegistry(None)
    reg.by_marker = {3: {"marker_id": "3", "name": "Reza", "employee_id": "E-3"}}
    frame, box, kxy, kc = _person_with_vest_marker()
    assert HelmetMarkerReader(reg).read(frame, box, kxy, kc) == "E-3"
    assert HelmetMarkerReader(reg, regions=["head"]).read(frame, box, kxy, kc) is None     # helmet only
    with pytest.raises(ValueError):
        HelmetMarkerReader(reg, regions=["legs"])


def test_marker_size_needed_for_a_camera():
    assert abs(required_marker_cm(8.0, 1920, 90.0) - 20.0) < 0.5        # 1080p, 90 deg, 8 m -> ~20 cm (vest)
    assert required_marker_cm(3.0, 1920, 90.0) < 8.0                     # helmet sticker only up close
    assert required_marker_cm(8.0, 3840, 90.0) < 11.0                    # 4K halves it


def test_make_markers_prints_the_size_and_writes_files(tmp_path, capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_markers

    assert make_markers.main(["--ids", "1-3", "--distance-m", "8", "--out", str(tmp_path)]) == 0
    assert len(list(tmp_path.glob("marker_*.png"))) == 3
    assert "vest" in capsys.readouterr().out


# ------------------------------------------------------------------ marker on a recovered binding
class FakeMarkers:
    def __init__(self):
        self.by_x = {}

    def read(self, frame, bbox, kxy, kconf):
        return self.by_x.get(int(round(float(bbox[0]))))


def _a_hides_behind_the_truck(markers, a_label_x=()):
    s = Site(marker_reader=markers)
    a = unit(0)
    b = sim_to(a, 1, 0.85)
    tb, ta = s.tid(), s.tid()
    for x in a_label_x:
        markers.by_x[x] = "E-7"
    s.run(6.0, lambda k: [(ta, 300.0 + 3.0 * (k % 10), a), (tb, 100.0, b)])
    gid_a = s.mgr.track_to_global[ta]
    t = s.tid()
    for k in range(16):
        s.step([(t, 330.0 + 20.0 * k, a), (tb, 100.0, b)])
    s.run(4.0, lambda k: [(tb, 100.0, b)])
    return s, a, b, tb, gid_a


def _emerge(s, emb, tb, b, frames=30):
    t = s.tid()
    out = {}
    for k in range(frames):
        out = s.step([(t, TRUCK_X2 + 5.0 + 6.0 * k, emb), (tb, 100.0, b)])
    return t, out.get(t)


def test_lookalike_with_a_marker_does_not_label_the_hidden_workers_history():
    markers = FakeMarkers()
    s, a, b, tb, gid_a = _a_hides_behind_the_truck(markers)          # A's marker was never read
    for k in range(30):
        markers.by_x[int(TRUCK_X2 + 5.0 + 6.0 * k)] = "E-8"           # B comes out, his vest is readable
    _, got = _emerge(s, sim_to(a, 20, 0.88), tb, b)
    assert got not in (None, gid_a) and s.mgr.identity_label(got) == "E-8"
    ident_a = s.mgr.identities[gid_a]
    assert ident_a.label is None                                     # A's history is NOT relabelled as B
    assert abs(float(ident_a.last_bbox_xyxy[0]) - 630.0) < 1.0       # A put back where he disappeared
    markers.by_x = {}
    _, got_a = _emerge(s, sim_to(a, 21, 0.88), tb, b)                 # the real A comes out later
    assert got_a == gid_a


def test_lookalike_with_another_marker_does_not_take_a_labelled_workers_place():
    markers = FakeMarkers()
    s, a, b, tb, gid_a = _a_hides_behind_the_truck(markers, a_label_x=[300 + 3 * k for k in range(10)])
    assert s.mgr.identity_label(gid_a) == "E-7"
    for k in range(30):
        markers.by_x[int(TRUCK_X2 + 5.0 + 6.0 * k)] = "E-8"
    _, got = _emerge(s, sim_to(a, 20, 0.88), tb, b)
    assert got != gid_a and s.mgr.identity_label(got) == "E-8"
    assert s.mgr.identity_label(gid_a) == "E-7"
    assert abs(float(s.mgr.identities[gid_a].last_bbox_xyxy[0]) - 630.0) < 1.0


def test_the_same_worker_reading_his_own_marker_keeps_his_id():
    markers = FakeMarkers()
    s, a, b, tb, gid_a = _a_hides_behind_the_truck(markers, a_label_x=[300 + 3 * k for k in range(10)])
    for k in range(30):
        markers.by_x[int(TRUCK_X2 + 5.0 + 6.0 * k)] = "E-7"
    _, got = _emerge(s, sim_to(a, 20, 0.88), tb, b)
    assert got == gid_a and s.mgr.identity_label(gid_a) == "E-7"
