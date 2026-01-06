import pytest
import numpy as np
from pathlib import Path
import sys
import os
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestSingleImageProcessing:
    def test_script_exists(self):
        assert Path('single_image_processing.py').exists()

    def test_script_can_be_imported(self):
        try:
            import single_image_processing as sip
            assert sip is not None
        except ImportError as e:
            pytest.fail(f"Failed to fetch single_image_processing.py: {e}")

    def test_load_image_functionality(self, temp_dir):
        try:
            import single_image_processing as sip

            test_img_path = temp_dir / 'test_mri.jpg'
            img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(test_img_path)

            if hasattr(sip, 'load_image'):
                loaded = sip.load_image(str(test_img_path))
                assert loaded is not None
                assert hasattr(loaded, 'shape') or isinstance(loaded, Image.Image)

        except ImportError:
            pytest.skip("single_image_processing.py failed to fetch")
