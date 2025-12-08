import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def train_random_forest(X, y):
    # Flatten (N, 6, 250) → (N, 1500)
    X_flat = X.reshape(X.shape[0], -1)

    print("Training Random Forest on:", X_flat.shape)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1,
        random_state=42
    )

    clf.fit(X_flat, y)

    y_pred = clf.predict(X_flat)

    print("\nRandom Forest Results (TRAIN SET ONLY):")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Macro F1:", f1_score(y, y_pred, average='macro'))
    print(classification_report(y, y_pred))

    return clf
