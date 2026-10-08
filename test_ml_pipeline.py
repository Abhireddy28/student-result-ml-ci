import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression


MODEL_FILE = "student_result_model.pkl"
METRICS_FILE = "metrics.pkl"


def test_model_file_exists():
    assert os.path.exists(MODEL_FILE)


def test_metrics_file_exists():
    assert os.path.exists(METRICS_FILE)


def test_model_loads():
    model = joblib.load(MODEL_FILE)
    assert model is not None
    assert isinstance(model, LogisticRegression)


def test_metrics_load():
    metrics = joblib.load(METRICS_FILE)
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics


def test_accuracy_range():
    metrics = joblib.load(METRICS_FILE)
    accuracy = metrics["accuracy"]

    assert 0.0 <= accuracy <= 1.0


def test_confusion_matrix_shape():
    metrics = joblib.load(METRICS_FILE)
    cm = np.array(metrics["confusion_matrix"])

    assert cm.shape == (2, 2)


def test_model_prediction():
    model = joblib.load(MODEL_FILE)

    sample = pd.DataFrame(
        [[85, 75, 80]],
        columns=[
            "attendance",
            "internal_marks",
            "assignment_score"
        ]
    )

    prediction = model.predict(sample)

    assert prediction[0] in [0, 1]
