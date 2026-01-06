import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize


def process_single_image(path: str) -> None:
    with open("model.p", "rb") as f:
        model = pickle.load(f)

    def preprocess_image(path: str) -> np.ndarray:
        img = imread(path)
        img = resize(img, (15, 15))
        return img.flatten().reshape(1, -1)

    image = preprocess_image(path)
    prediction = model.predict(image)

    print("Prediction:", "Tumor" if prediction[0] == 0 else "Normal")


process_single_image("images/Validation/Tumor/Tumor (1).jpg")
