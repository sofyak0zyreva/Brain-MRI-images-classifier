import pickle
import sys
import numpy as np
from skimage.io import imread
from skimage.transform import resize


def process_single_image(path: str) -> None:
    """
    A function for making prediction for a single image based on pre-made model from `model.p` file
    """
    with open("model.p", "rb") as f:
        model = pickle.load(f)

    def preprocess_image(path: str) -> np.ndarray:
        img = imread(path)
        img = resize(img, (15, 15))
        return img.flatten().reshape(1, -1)

    image = preprocess_image(path)
    prediction = model.predict(image)

    print("Prediction:", "Tumor" if prediction[0] == 0 else "Normal")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    process_single_image(image_path)
