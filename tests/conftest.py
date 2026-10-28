"""Shared fixtures for the host-tool tests (run with the lerobot venv, offline, CPU only).

    HF_HUB_OFFLINE=1 <venv>/bin/python -m pytest tests -q

Session fixtures build the expensive artefacts once: synthetic bags -> dataset -> BC run.
"""

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
TESTS = Path(__file__).resolve().parent
JBL_NODES = REPO / "ros2_ws" / "src" / "jbl_nodes"  # shared preprocessing and the robot's policy loader
for _path in (str(TOOLS), str(TESTS), str(JBL_NODES)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("SVT_LOG", "1")  # keep the AV1 encoder's per-episode banner out of the logs

REPO_ID = "local/jbl_synthetic"
FPS = 10


@pytest.fixture(scope="session")
def bags_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("bags")


@pytest.fixture(scope="session")
def synthetic_bags(bags_dir):
    """name -> RecordedBag for make_synthetic_bags.default_specs()."""
    import make_synthetic_bags

    return make_synthetic_bags.write_default_bags(bags_dir)


@pytest.fixture(scope="session")
def converted(synthetic_bags, bags_dir, tmp_path_factory):
    """(dataset root, ConversionReport) for all synthetic bags, default options (video)."""
    import bag_to_lerobot

    root = tmp_path_factory.mktemp("datasets") / "jbl_synthetic"
    report = bag_to_lerobot.convert(bag_to_lerobot.find_bags([bags_dir]), root, REPO_ID, fps=FPS)
    return root, report


@pytest.fixture(scope="session")
def dataset_root(converted):
    return converted[0]


@pytest.fixture(scope="session")
def bc_run(dataset_root, tmp_path_factory):
    """Output directory of a 2-epoch CPU train_bc run on the synthetic dataset."""
    import train_bc

    out = tmp_path_factory.mktemp("bc_run")
    rc = train_bc.main([
        "--root", str(dataset_root), "--repo-id", REPO_ID, "--out", str(out),
        "--epochs", "2", "--batch-size", "16", "--num-workers", "0", "--device", "cpu",
        "--val-fraction", "0.25", "--seed", "0",
    ])
    assert rc == 0
    return out
