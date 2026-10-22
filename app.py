from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request, g, Response
import logging
import sys
import time
import uuid

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("student-result-api")

# ------------------------------------------------------------
# Prometheus Metrics
# ------------------------------------------------------------
HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the model is loaded (1=yes, 0=no)"
)

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")
FEATURES = ["attendance", "internal_marks", "assignment_score"]

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        MODEL_LOADED.set(1)
        logger.info("event=model_loaded model_file=%s", MODEL_PATH)
        return model
    except Exception:
        MODEL_LOADED.set(0)
        logger.exception("event=model_load_failed model_file=%s", MODEL_PATH)
        raise

model = load_model()

# ------------------------------------------------------------
# Request middleware (logging + metrics)
# ------------------------------------------------------------
@app.before_request
def before_request():
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    g.start_time = time.perf_counter()

@app.after_request
def after_request(response):
    duration = time.perf_counter() - g.start_time
    path = request.path
    method = request.method
    status = str(response.status_code)

    # Do not count /metrics scrapes as user traffic
    if path != "/metrics":
        HTTP_REQUESTS.labels(method=method, path=path, status=status).inc()
        HTTP_LATENCY.labels(method=method, path=path).observe(duration)

    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        g.request_id, method, path, status, duration * 1000
    )
    response.headers["X-Request-ID"] = g.request_id
    return response

# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/")
def health_check():
    return jsonify({"status": "ok", "service": "student-result-prediction"})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        logger.warning("event=validation_error path=/predict reason=missing_json_body")
        return jsonify({"error": "JSON request body is required"}), 400

    missing = [f for f in FEATURES if f not in data]
    if missing:
        logger.warning("event=validation_error path=/predict missing_fields=%s", missing)
        return jsonify({"error": "Missing required fields", "missing_fields": missing}), 400

    try:
        logger.info("event=prediction_request feature_names=%s", FEATURES)
        sample = pd.DataFrame([{f: data[f] for f in FEATURES}])
        prediction_code = int(model.predict(sample)[0])
        prediction = "PASS" if prediction_code == 1 else "FAIL"

        PREDICTIONS.labels(result="success").inc()
        logger.info("event=prediction_success prediction=%s", prediction)

        return jsonify({"prediction": prediction, "prediction_code": prediction_code})

    except Exception:
        PREDICTIONS.labels(result="error").inc()
        logger.exception("event=prediction_failed")
        return jsonify({"error": "Prediction failed"}), 500

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
