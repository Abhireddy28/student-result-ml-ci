from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request, g
import logging
import sys
import time
import uuid

# ------------------------------------------------------------
# Logging configuration (writes to stdout so Render can capture it)
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("student-result-api")

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_score"
]

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("event=model_loaded model_file=%s", MODEL_PATH)
        return model
    except Exception:
        logger.exception("event=model_load_failed model_file=%s", MODEL_PATH)
        raise

# Load model at startup
model = load_model()

# ------------------------------------------------------------
# Request logging middleware (before + after request)
# ------------------------------------------------------------
@app.before_request
def before_request():
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    g.start_time = time.perf_counter()

@app.after_request
def after_request(response):
    duration_ms = (time.perf_counter() - g.start_time) * 1000
    level = logging.WARNING if response.status_code >= 400 else logging.INFO

    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-ID"] = g.request_id
    return response

# ------------------------------------------------------------
# Health endpoint
# ------------------------------------------------------------
@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })

# ------------------------------------------------------------
# Prediction endpoint with logging
# ------------------------------------------------------------
@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        logger.warning("event=validation_error path=/predict reason=missing_json_body")
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [f for f in FEATURES if f not in data]
    if missing_fields:
        logger.warning(
            "event=validation_error path=/predict missing_fields=%s",
            missing_fields
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        logger.info(
            "event=prediction_request feature_names=%s",
            FEATURES
        )

        sample = pd.DataFrame([{f: data[f] for f in FEATURES}])
        prediction_code = int(model.predict(sample)[0])
        prediction = "PASS" if prediction_code == 1 else "FAIL"

        logger.info(
            "event=prediction_success prediction=%s prediction_code=%s",
            prediction,
            prediction_code
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except Exception:
        logger.exception("event=prediction_failed")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
