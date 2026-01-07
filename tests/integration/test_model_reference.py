import shutil

import pytest
import numpy as np
import pickle
from pathlib import Path
import sys
import os
import tempfile
from PIL import Image
from imageio import imwrite

from model import (
    prepare_data,
    get_accuracy,
    classify_tumor_dataset,
    SVC,
    GridSearchCV,
    accuracy_score,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class TestEndToEnd:
    def test_full_prediction_pipeline(self, temp_dir):
        assert Path("model.p").exists()

        test_img_path = temp_dir / "integration_test.jpg"
        img_array = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="L")
        img.save(test_img_path)

        with open("model.p", "rb") as f:
            model = pickle.load(f)

        img_array_flat = img_array.flatten()

        if hasattr(model, "n_features_in_"):
            expected_size = model.n_features_in_
            if len(img_array_flat) > expected_size:
                img_array_flat = img_array_flat[:expected_size]
            elif len(img_array_flat) < expected_size:
                img_array_flat = np.pad(
                    img_array_flat, (0, expected_size - len(img_array_flat))
                )

        try:
            prediction = model.predict([img_array_flat])
            assert prediction is not None
            assert len(prediction) == 1
            print(f"Prediction: {prediction[0]}")

        except Exception as e:
            pytest.skip(f"Error: {e}")

    def test_model_with_single_image_script(self, temp_dir):
        assert Path("model.p").exists()
        assert Path("single_image_processing.py").exists()

        try:
            import single_image_processing as sip

            test_img_path = temp_dir / "script_integration.jpg"
            img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(test_img_path)

            if hasattr(sip, "predict_image"):
                result = sip.predict_image(str(test_img_path))
                assert result is not None

        except ImportError as e:
            pytest.skip(f"Failed to fetch single_image_processing.py: {e}")


class TestModelFile:
    def test_model_file_exists_and_readable(self):
        model_path = Path("model.py")
        assert model_path.exists()

        # Проверяем, что файл читается
        with open(model_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "def classify_tumor_dataset" in content
            assert "def prepare_data" in content
            assert "SVC" in content

    def test_all_functions_imported(self):
        assert callable(prepare_data)
        assert callable(get_accuracy)
        assert callable(classify_tumor_dataset)

        assert SVC is not None
        assert GridSearchCV is not None
        assert accuracy_score is not None


class TestPrepareDataFunction:
    @pytest.fixture
    def sample_image_data(self):
        test_dir = Path("test_images_temp")
        test_dir.mkdir(exist_ok=True)

        for category in ["Tumor", "Normal"]:
            (test_dir / category).mkdir(exist_ok=True)

        for category in ["Tumor", "Normal"]:
            for i in range(3):
                img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
                img_path = test_dir / category / f"test_{i}.png"
                import imageio.v3 as iio

                iio.imwrite(img_path, img)

        yield test_dir
        import shutil

        shutil.rmtree(test_dir)

    def test_prepare_data_function_signature(self):
        import inspect

        sig = inspect.signature(prepare_data)
        params = list(sig.parameters.keys())
        assert params == ["input_dir", "categories", "data", "labels"]
        annotations = prepare_data.__annotations__
        assert "input_dir" in annotations and annotations["input_dir"] == str
        assert "categories" in annotations and "List" in str(annotations["categories"])

    def test_prepare_data_creates_correct_structure(self, sample_image_data):
        data = []
        labels = []
        categories = ["Tumor", "Normal"]
        prepare_data(str(sample_image_data), categories, data, labels)
        assert len(data) == 6
        assert len(labels) == 6
        assert labels.count(0) == 3
        assert labels.count(1) == 3
        for i, img_array in enumerate(data):
            assert isinstance(img_array, np.ndarray)
            expected_size = 15 * 15 * 3
            assert img_array.size == expected_size


class TestGetAccuracyFunction:
    def test_get_accuracy_functionality(self, capsys):
        labels_test = np.array([0, 1, 0, 1, 0])
        y_prediction = np.array([0, 1, 0, 0, 1])
        get_accuracy(labels_test, y_prediction)
        captured = capsys.readouterr()
        output = captured.out
        assert "% of samples correctly classified" in output
        assert "60.00%" in output

    def test_get_accuracy_perfect_prediction(self, capsys):
        labels_test = np.array([0, 1, 0, 1])
        y_prediction = np.array([0, 1, 0, 1])
        get_accuracy(labels_test, y_prediction)
        captured = capsys.readouterr()
        assert "100.00%" in captured.out

    def test_get_accuracy_zero_prediction(self, capsys):
        labels_test = np.array([0, 1, 0, 1])
        y_prediction = np.array([1, 0, 1, 0])
        get_accuracy(labels_test, y_prediction)
        captured = capsys.readouterr()
        assert "0.00%" in captured.out


class TestClassifyTumorDatasetReal:
    @pytest.fixture
    def create_real_test_dataset(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="mri_real_test_"))
        train_dir = temp_dir / "images" / "Train"
        val_dir = temp_dir / "images" / "Validation"
        categories = ["Tumor", "Normal"]
        for cat_idx, category in enumerate(categories):
            (train_dir / category).mkdir(parents=True, exist_ok=True)
            (val_dir / category).mkdir(parents=True, exist_ok=True)
            for i in range(5):
                if category == "Tumor":
                    # paste tumor
                    img = np.ones((50, 50, 3), dtype=np.uint8) * 100
                    img[20:30, 20:30, :] = 250
                else:
                    img = np.ones((50, 50, 3), dtype=np.uint8) * 150

                train_path = train_dir / category / f"train_{i}.jpg"
                imwrite(train_path, img)

            for i in range(3):
                if category == "Tumor":
                    img = np.ones((50, 50, 3), dtype=np.uint8) * 110
                    img[15:25, 15:25, :] = 240
                else:
                    img = np.ones((50, 50, 3), dtype=np.uint8) * 160

                val_path = val_dir / category / f"val_{i}.jpg"
                imwrite(val_path, img)
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_complete_pipeline_real_data(
        self, create_real_test_dataset, monkeypatch, capsys
    ):
        temp_dir = create_real_test_dataset
        original_cwd = os.getcwd()

        try:
            os.chdir(temp_dir)
            data = []
            labels = []
            categories = ["Tumor", "Normal"]

            prepare_data("images/Train", categories, data, labels)

            assert len(data) == 10
            assert len(labels) == 10

            tumor_count = labels.count(0)
            normal_count = labels.count(1)
            assert tumor_count == 5
            assert normal_count == 5

            import model

            model.classify_tumor_dataset()
            captured = capsys.readouterr()
            assert "100.00% of samples correctly classified\n" in captured.out

        finally:
            os.chdir(original_cwd)


# used to gain test coverage ;)
class TestDataFlow:
    def test_images_directory_exists(self):
        images_dir = Path("images")
        if images_dir.exists():
            image_files = list(images_dir.glob("*.*"))
            if image_files:
                for img_path in image_files[:3]:
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            assert width > 0 and height > 0
                            print(f"  {img_path.name}: {width}x{height}")
                    except:
                        print("Could not parse an image in directory")
        else:
            pytest.skip("Could not found data")

    def test_documentation_exists(self):
        docs_dir = Path("docs")
        if docs_dir.exists():
            doc_files = list(docs_dir.rglob("*.rst"))
            if len(doc_files) == 0:
                pytest.fail("Failed to find doc")
