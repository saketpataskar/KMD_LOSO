import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

def train_svm(X, y):
    # Flatten (N, 6, 250) → (N, 1500)
    X_flat = X.reshape(X.shape[0], -1)

    print("Training SVM on:", X_flat.shape)

    clf = SVC(
        C=10,
        kernel='rbf',
        gamma='scale'
    )

    clf.fit(X_flat, y)

    y_pred = clf.predict(X_flat)

    print("\nSVM Results (TRAIN SET ONLY):")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Macro F1:", f1_score(y, y_pred, average='macro'))
    print(classification_report(y, y_pred))

    return clf
