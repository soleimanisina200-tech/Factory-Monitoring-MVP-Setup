"""Shared pytest setup: puts the repo root on sys.path and stubs heavy optional deps."""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# torchreid is only needed by the real OSNet extractor. Unit tests inject a fake
# extractor, so a stub keeps `import app.identity.reid_model` importable on machines
# (CI, laptops) where torchreid is not installed.
try:  # pragma: no cover - depends on the environment
    import torchreid  # noqa: F401
except Exception:  # pragma: no cover
    stub = types.ModuleType("torchreid")
    stub.utils = types.SimpleNamespace(FeatureExtractor=object)
    sys.modules["torchreid"] = stub
