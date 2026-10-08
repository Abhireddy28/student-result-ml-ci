import numpy as np
import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# Generate student data
np.random.seed(42)

data = {
    "attendance": np.random.randint(50, 101, 300),
    "internal_marks": np.random.randint(30, 101, 300),
    "assignment_score": np.random.randint(30, 101, 300)
}

df = pd.DataFrame(data)
df.to_csv("student_results.csv", index=False)
print("Dataset saved as student_results.csv")

# Create result
df["result"] = (
    (df["attendance"] >= 75) &
    (df["internal_marks"] >= 50) &
    (df["assignment_score"] >= 50)
).astype(int)


# Features and target
X = df[["attendance", "internal_marks", "assignment_score"]]
y = df["result"]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)


# Save model
joblib.dump(model, "student_result_model.pkl")

# Save metrics as JSON (required by quality_gate.py)
metrics = {
    "accuracy": float(accuracy),
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Model and metrics saved successfully.")
