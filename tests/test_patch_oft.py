import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("patch_oft", ROOT / "scripts" / "patch_openvla_oft.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_patch_openvla_oft_idempotent(tmp_path):
    base = tmp_path / "prismatic" / "vla" / "datasets" / "rlds" / "oxe"
    base.mkdir(parents=True)
    (base / "configs.py").write_text('OXE_DATASET_CONFIGS = {"libero_spatial_no_noops": {"a": 1}}\n')
    (base / "transforms.py").write_text(
        "def libero_dataset_transform(t):\n    return t\nOXE_STANDARDIZATION_TRANSFORMS = {}\n"
    )
    (base / "mixtures.py").write_text("OXE_NAMED_MIXTURES = {}\n")
    m = _load()
    assert len(m.patch(tmp_path)) == 3
    assert m.patch(tmp_path) == []  # second run is a no-op
    ns = {}
    for f in ("configs.py", "transforms.py", "mixtures.py"):
        src = (base / f).read_text()
        ast.parse(src)
        exec(src, ns)
    assert ns["OXE_DATASET_CONFIGS"]["occlubench_libero"] == {"a": 1}
    assert ns["OXE_STANDARDIZATION_TRANSFORMS"]["occlubench_libero"] is ns["libero_dataset_transform"]
    assert ns["OXE_NAMED_MIXTURES"]["occlubench_libero"] == [("occlubench_libero", 1.0)]
