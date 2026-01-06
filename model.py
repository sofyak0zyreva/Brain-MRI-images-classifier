import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

input_dir = 'images/Train'
categories = ['Tumor', 'Normal']

data = []
labels = []

for category_indx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_indx)

data = np.asarray(data)
labels = np.asarray(labels)

test_input_dir = 'images/Validation'
test_categories = ['Tumor', 'Normal']

test_data = []
test_labels = []

for category_indx, category in enumerate(test_categories):
    for file in os.listdir(os.path.join(test_input_dir, category)):
        img_path = os.path.join(test_input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        test_data.append(img.flatten())
        test_labels.append(category_indx)

test_data = np.asarray(test_data)
test_labels = np.asarray(test_labels)


classifier = SVC()
parameters = {'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}

grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(data, labels)

best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(test_data)
score = accuracy_score(y_prediction, test_labels)

print(f"{score*100}% of samples correctly classified")
