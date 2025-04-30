import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to evaluate SVM with a given kernel
def evaluate_svm(kernel, **kwargs):
    clf = SVC(kernel=kernel, **kwargs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\nKernel: {kernel.capitalize()}")
    if 'degree' in kwargs:
        print(f"Degree: {kwargs['degree']}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Evaluate SVM with different kernels
kernels = [
    ('linear', {}),
    ('poly', {'degree': 3}),
    ('rbf', {}),
    ('sigmoid', {})
]

for kernel, params in kernels:
    evaluate_svm(kernel, **params)