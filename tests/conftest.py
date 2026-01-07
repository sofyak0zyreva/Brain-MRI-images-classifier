import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def temp_dir():
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_image_array():
    return np.random.rand(128, 128).astype(np.float32)


@pytest.fixture
def sample_rgb_image_array():
    return np.random.rand(128, 128, 3).astype(np.float32)


@pytest.fixture
def sample_labels():
    return ["normal", "tumor"]


@pytest.fixture
def sample_brain_mri_paths():
    return [
        "tests/data/sample_mri/normal_sample.jpg",
        "tests/data/sample_mri/tumor_sample.jpg",
    ]
