
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_PATH = Path(__file__).resolve().parents[2]
ARTIFACTS_PATH = BASE_PATH / "artifacts"

MODEL_PATH = ARTIFACTS_PATH / "model.onnx"
CLASSES_PATH = ARTIFACTS_PATH / "classes.json"
BREED_TYPES_PATH = ARTIFACTS_PATH / "breed_animal_types.json"

MODEL_VERSION = "v3"
CONFIDENCE_THRESHOLD = 0.80

app = FastAPI(title="Pets API", version=MODEL_VERSION)

# --------------------------------------------------
# Lightweight Prometheus metrics (no extra Python dependency)
# --------------------------------------------------

import threading
import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import PlainTextResponse

_metrics_lock = threading.Lock()
_request_counts = defaultdict(int)
_request_duration_sum = defaultdict(float)
_request_duration_count = defaultdict(int)
_prediction_decisions = defaultdict(int)
_confidence_buckets = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
_confidence_bucket_counts = [0 for _ in _confidence_buckets]
_confidence_sum = 0.0
_confidence_count = 0


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - started
        route = request.url.path
        with _metrics_lock:
            key = (request.method, route, status_code)
            _request_counts[key] += 1
            _request_duration_sum[(request.method, route)] += elapsed
            _request_duration_count[(request.method, route)] += 1


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Expose service and prediction metrics in Prometheus text format."""
    lines = [
        "# HELP pets_api_http_requests_total Total HTTP requests handled.",
        "# TYPE pets_api_http_requests_total counter",
    ]
    with _metrics_lock:
        for (method, route, status), count in sorted(_request_counts.items()):
            lines.append(
                f'pets_api_http_requests_total{{method="{method}",'
                f'path="{route}",status="{status}"}} {count}'
            )

        lines.extend([
            "# HELP pets_api_http_request_duration_seconds_sum Total HTTP request duration in seconds.",
            "# TYPE pets_api_http_request_duration_seconds_sum counter",
        ])
        for (method, route), value in sorted(_request_duration_sum.items()):
            lines.append(
                f'pets_api_http_request_duration_seconds_sum{{method="{method}",'
                f'path="{route}"}} {value:.9f}'
            )

        lines.extend([
            "# HELP pets_api_http_request_duration_seconds_count Number of timed HTTP requests.",
            "# TYPE pets_api_http_request_duration_seconds_count counter",
        ])
        for (method, route), count in sorted(_request_duration_count.items()):
            lines.append(
                f'pets_api_http_request_duration_seconds_count{{method="{method}",'
                f'path="{route}"}} {count}'
            )

        lines.extend([
            "# HELP pets_api_predictions_total Successful predictions by decision.",
            "# TYPE pets_api_predictions_total counter",
        ])
        for decision, count in sorted(_prediction_decisions.items()):
            lines.append(
                f'pets_api_predictions_total{{decision="{decision}"}} {count}'
            )

        lines.extend([
            "# HELP pets_api_prediction_confidence Prediction confidence distribution.",
            "# TYPE pets_api_prediction_confidence histogram",
        ])
        cumulative = 0
        for bound, count in zip(_confidence_buckets, _confidence_bucket_counts):
            cumulative += count
            lines.append(
                f'pets_api_prediction_confidence_bucket{{le="{bound}"}} {cumulative}'
            )
        lines.extend([
            f'pets_api_prediction_confidence_bucket{{le="+Inf"}} {_confidence_count}',
            f'pets_api_prediction_confidence_sum {_confidence_sum:.9f}',
            f'pets_api_prediction_confidence_count {_confidence_count}',
        ])

    return PlainTextResponse(
        "\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


def record_prediction_metrics(confidence: float, decision: str) -> None:
    global _confidence_sum, _confidence_count
    with _metrics_lock:
        _prediction_decisions[decision] += 1
        _confidence_sum += confidence
        _confidence_count += 1
        for index, bound in enumerate(_confidence_buckets):
            if confidence <= bound:
                _confidence_bucket_counts[index] += 1
                break


# --------------------------------------------------
# Load model and class mapping
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

if not CLASSES_PATH.exists():
    raise FileNotFoundError(f"Class mapping not found: {CLASSES_PATH}")

with open(BREED_TYPES_PATH, encoding="utf-8") as f:
    BREED_ANIMAL_TYPES = json.load(f)


model = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

input_meta = model.get_inputs()[0]
output_meta = model.get_outputs()[0]

INPUT_NAME = input_meta.name
OUTPUT_NAME = output_meta.name

# This implementation expects an NCHW model with 3 channels.
if len(input_meta.shape) != 4 or input_meta.shape[1] != 3:
    raise ValueError(
        f"Expected an NCHW image input, got: {input_meta.shape}"
    )

with open(CLASSES_PATH, encoding="utf-8") as f:
    class_to_idx = json.load(f)

if not isinstance(class_to_idx, dict) or not class_to_idx:
    raise ValueError("classes.json must contain a non-empty class mapping.")

# Convert JSON indices to integers and order classes by model output index.
indexed_classes = {
    int(index): name
    for name, index in class_to_idx.items()
}

if sorted(indexed_classes) != list(range(len(indexed_classes))):
    raise ValueError("Class indices must be consecutive, starting at 0.")

CLASS_NAMES = [
    indexed_classes[index]
    for index in range(len(indexed_classes))
]

if isinstance(output_meta.shape[-1], int) and output_meta.shape[-1] != len(CLASS_NAMES):
    raise ValueError(
        f"Model outputs {output_meta.shape[-1]} classes, "
        f"but classes.json contains {len(CLASS_NAMES)}."
    )


def normalize_breed_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())

BREED_TO_SPECIES = {
    normalize_breed_name(breed): species
    for breed, species in BREED_ANIMAL_TYPES.items()
}

def get_species(breed: str) -> str:
    normalized = normalize_breed_name(breed)

    if normalized not in BREED_TO_SPECIES:
        raise ValueError(f"Unknown breed: {breed}")

    return BREED_TO_SPECIES[normalized]

for breed in CLASS_NAMES:
    get_species(breed)


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    shape = input_meta.shape

    height = shape[2]
    width = shape[3]

    if not isinstance(height, int) or not isinstance(width, int):
        raise ValueError("The ONNX model requires fixed image dimensions.")

    image = image.convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    array = (array - mean) / std

    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)

    return np.ascontiguousarray(array, dtype=np.float32)


# --------------------------------------------------
# API response schema
# --------------------------------------------------

class TopPrediction(BaseModel):
    breed: str
    confidence: float


class PredictionResponse(BaseModel):
    breed: str
    species: str
    confidence: float
    top_3: list[TopPrediction]
    decision: str
    model_version: str


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "model": MODEL_PATH.name,
        "num_classes": len(CLASS_NAMES),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Please upload an image file.",
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty image file.")

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be 10 MB or smaller.",
            )

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_data = preprocess_image(image)

        outputs = model.run(
            [OUTPUT_NAME],
            {INPUT_NAME: input_data},
        )

        scores = np.asarray(outputs[0][0], dtype=np.float64)

        if scores.ndim != 1 or len(scores) != len(CLASS_NAMES):
            raise HTTPException(
                status_code=500,
                detail="Unexpected model output shape.",
            )

        if not np.all(np.isfinite(scores)):
            raise HTTPException(
                status_code=500,
                detail="Model produced invalid scores.",
            )

        shifted = scores - np.max(scores)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()

        top_indices = np.argsort(probabilities)[-3:][::-1]
        best_idx = int(top_indices[0])

        breed = CLASS_NAMES[best_idx]
        confidence = float(probabilities[best_idx])

        top_3 = [
            TopPrediction(
                breed=CLASS_NAMES[int(index)],
                confidence=round(float(probabilities[index]), 4),
            )
            for index in top_indices
        ]

        decision = (
            "confident"
            if confidence >= CONFIDENCE_THRESHOLD
            else "uncertain"
        )

        record_prediction_metrics(confidence, decision)

        return PredictionResponse(
            breed=breed,
            species=get_species(breed),
            confidence=round(confidence, 4),
            top_3=top_3,
            decision=decision,
            model_version=MODEL_VERSION,
        )

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc
