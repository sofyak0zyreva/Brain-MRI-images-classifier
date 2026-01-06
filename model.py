from typing import List
import numpy as np
import os
import pickle
from skimage.io import imread  # type: ignore
from skimage.transform import resize  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore


def prepare_data(
    input_dir: str, categories: List[str], data: List[np.ndarray], labels: List[int]
) -> None:
    for category_indx, category in enumerate(categories):
        for file in os.listdir(os.path.join(input_dir, category)):
            img_path = os.path.join(input_dir, category, file)
            img = imread(img_path)
            img = resize(img, (15, 15))
            data.append(img.flatten())
            labels.append(category_indx)
    # data = np.asarray(data)
    # labels = np.asarray(labels)


input_dir = "images/Train"
categories = ["Tumor", "Normal"]

data: List[np.ndarray] = []
labels: List[int] = []

prepare_data(input_dir, categories, data, labels)

test_input_dir = "images/Validation"
test_categories = ["Tumor", "Normal"]

test_data: List[np.ndarray] = []
test_labels: List[int] = []

prepare_data(test_input_dir, test_categories, test_data, test_labels)

data_train = np.asarray(data)
labels_train = np.asarray(labels)
data_test = np.asarray(test_data)
labels_test = np.asarray(test_labels)

classifier = SVC()
parameters = {
    "gamma": [0.01, 0.001, 0.0001],
    "C": [1, 10, 100, 1000],
}

grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(data_train, labels_train)

best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(data_test)

score = accuracy_score(labels_test, y_prediction)
print(f"{score * 100:.2f}% of samples correctly classified")

with open("model.p", "wb") as f:
    pickle.dump(best_estimator, f)
