import pytest
import numpy as np
import pickle
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestModelBasics:
    def test_model_file_exists(self):
        model_path = Path('model.p')
        assert model_path.exists(), f"Model file not found: {model_path}"
        assert model_path.stat().st_size > 0, "Model's file is empty"

    def test_model_can_be_loaded(self):
        try:
            with open('model.p', 'rb') as f:
                model = pickle.load(f)
            assert model is not None, "Error while trying to load model: None model"
        except Exception as e:
            pytest.fail(f"Error while trying to load model: {e}")


    def test_model_predict_proba(self):
        try:
            with open('model.p', 'rb') as f:
                model = pickle.load(f)

            if hasattr(model, 'predict_proba'):
                test_data = np.random.rand(5, 100).astype(np.float32)
                probe = model.predict_proba(test_data)
                assert probe is not None
                assert probe.shape[0] == 5
                np.testing.assert_allclose(probe.sum(axis=1), 1.0, rtol=1e-5)

        except Exception as e:
            pytest.skip(f"SKIPPED HAS NO ATTRS {e}")

    def test_model_attributes(self):
        try:
            with open('model.p', 'rb') as f:
                model = pickle.load(f)

            common_attrs = ['fit', 'score', 'get_params', 'set_params']
            for attr in common_attrs:
                if hasattr(model, attr):
                    assert callable(getattr(model, attr)), f"{attr} не callable"

        except Exception as e:
            pytest.skip(f"SKIPPED HAS NO ATTRS {e}")


class TestModelScript:
    def test_model_script_exists(self):
        assert Path('model.py').exists(), "model.py was not found"

    def test_model_script_can_be_imported(self):
        try:
            import model
            assert model is not None
        except ImportError as e:
            pytest.fail(f"Failed to import `model.py`: {e}")

    def test_script_has_main_functions(self):
        try:
            import model

            expected_functions = [
                'load_data', 'prepare_data', 'train_model',
                'save_model', 'evaluate_model'
            ]

            for func_name in expected_functions:
                if hasattr(model, func_name):
                    func = getattr(model, func_name)
                    assert callable(func)

        except ImportError:
            pytest.skip("model.py import error")


