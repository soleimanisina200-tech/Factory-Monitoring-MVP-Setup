"""Re-ID backbone selection: the right checkpoint for the right model, never a silent mismatch."""
import pytest
import torch

from app.identity import reid_model as rm


class Small(torch.nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, width, 3)
        self.bn = torch.nn.BatchNorm2d(width)
        self.classifier = torch.nn.Linear(width, 10)


def test_match_ratio_detects_wrong_architecture_and_ignores_classifier():
    a, b = Small(8), Small(16)
    assert rm.weights_match_ratio(a, a.state_dict()) == 1.0
    assert rm.weights_match_ratio(a, b.state_dict()) < 0.2  # only BatchNorm's scalar counter has the same shape
    # Classifier differences (different training identities) must not matter; "module." prefixes are stripped.
    sd = {f"module.{k}": v for k, v in a.state_dict().items() if not k.startswith("classifier")}
    assert rm.weights_match_ratio(a, {"state_dict": sd}) == 1.0


def _offline(*_a, **_k):
    raise OSError("no network in tests")


def test_missing_weights_that_cannot_be_downloaded_stop_with_instructions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rm, "_download_gdrive", _offline)
    monkeypatch.setattr(rm, "_download_http", _offline)
    with pytest.raises(RuntimeError, match="osnet_ain_x1_0"):
        rm.resolve_reid_weights("osnet_ain_x1_0", None)
    with pytest.raises(RuntimeError, match="Failed to acquire MSMT17 weights"):
        rm.resolve_reid_weights("osnet_x1_0", None)


def test_mirror_is_used_when_google_drive_refuses(tmp_path, monkeypatch):
    """Google Drive often answers 'too many accesses': the first start must not depend on it."""
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_http(url, path, timeout=60.0):
        calls.append(url)
        path.write_bytes(b"0" * (rm.MIN_WEIGHT_BYTES + 1))

    monkeypatch.setattr(rm, "_download_gdrive", _offline)
    monkeypatch.setattr(rm, "_download_http", fake_http)
    got = rm.resolve_reid_weights("osnet_x1_0", None)
    assert got.replace("\\", "/") == "weights/osnet_x1_0_msmt17.pt"
    assert len(calls) == 1 and calls[0].startswith(rm.HF_MIRROR_BASE) and "osnet_x1_0_msmt17" in calls[0]


def test_truncated_download_is_not_accepted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rm, "_download_gdrive", _offline)
    monkeypatch.setattr(rm, "_download_http", lambda url, path, timeout=60.0: path.write_bytes(b"<html>error"))
    with pytest.raises(RuntimeError):
        rm.resolve_reid_weights("osnet_x1_0", None)
    assert not (tmp_path / "weights" / "osnet_x1_0_msmt17.pt").exists()


def test_explicit_existing_weights_path_is_used(tmp_path):
    f = tmp_path / "w.pt"
    f.write_bytes(b"x")
    assert rm.resolve_reid_weights("osnet_ain_x1_0", str(f)) == str(f)


def test_default_path_is_used_when_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "osnet_ain_x1_0_msmt17.pt").write_bytes(b"x")
    assert rm.resolve_reid_weights("osnet_ain_x1_0", None).replace("\\", "/") == "weights/osnet_ain_x1_0_msmt17.pt"


def test_real_osnet_weights_do_not_fit_osnet_ain():
    """With the real torchreid architectures: OSNet weights cover only a small part of OSNet-AIN."""
    torchreid = rm.torchreid
    if not hasattr(torchreid, "models"):
        pytest.skip("torchreid not importable in this environment")
    plain = torchreid.models.build_model("osnet_x1_0", num_classes=1, pretrained=False)
    ain = torchreid.models.build_model("osnet_ain_x1_0", num_classes=1, pretrained=False)
    assert rm.weights_match_ratio(ain, plain.state_dict()) < 0.5
    assert rm.weights_match_ratio(ain, ain.state_dict()) == 1.0


def test_verify_rejects_mismatching_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(rm.torchreid, "models", type("M", (), {"build_model": staticmethod(lambda **kw: Small(8))}),
                        raising=False)
    good, bad = tmp_path / "good.pt", tmp_path / "bad.pt"
    torch.save(Small(8).state_dict(), good)
    torch.save(Small(16).state_dict(), bad)
    assert rm.verify_reid_weights("osnet_x1_0", str(good)) == 1.0
    with pytest.raises(ValueError, match="do not fit"):
        rm.verify_reid_weights("osnet_x1_0", str(bad))
