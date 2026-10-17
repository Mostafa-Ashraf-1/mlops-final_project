# Pet Breed Classification — MLOps Pipeline

An end-to-end MLOps project that classifies images into **37 cat and dog breeds** using the Oxford-IIIT Pet dataset. It combines PyTorch training and MLflow tracking, a versioned data workflow, and a FastAPI prediction service powered by an exported ONNX model running on CPU. The repository also includes automated tests and a corruption-based input-drift evaluation suite.

## Get a prediction in 3 commands

Requirements: Docker with the Compose plugin, and a local pet image such as `pet.jpg`.

1. **Build the API image** (from the repository root):

   ```bash
   docker build -t pets-api .
   ```

2. **Start the API:**

   ```bash
   docker run --rm -d --name pets-api -p 8000:8000 pets-api
   ```

3. **Send an image to the model:**

   ```bash
   curl -X POST "http://localhost:8000/predict" -F "file=@pet.jpg"
   ```

Replace `pet.jpg` with the path to an image on your machine. A successful response contains the predicted breed and species, confidence, top-three predictions, a `confident` / `uncertain` decision, and model version. The API uses a confidence threshold of `0.80`; this is an inference service and does not train the model at startup.

Check service health at [http://localhost:8000/health](http://localhost:8000/health) or open the interactive API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

To stop the detached container after you finish:

```bash
docker stop pets-api
```

## What it does

- **Image classification:** trains transfer-learning experiments with ResNet-18, ResNet-50, and MobileNetV3-Small using PyTorch.
- **Experiment tracking:** records training and validation metrics in MLflow, including loss, top-1 accuracy, macro-F1, coverage, selective accuracy, and expected calibration error (ECE).
- **Model serving:** serves the exported ONNX model through FastAPI and ONNX Runtime on CPU.
- **Service monitoring:** exposes Prometheus metrics for HTTP request counts, request duration, prediction confidence, and confident/uncertain decisions; a provisioned Grafana dashboard visualizes these metrics.
- **Input validation and confidence handling:** returns the predicted breed, cat/dog species, confidence, top-three predictions, model version, and a confidence-based `confident` / `uncertain` decision.
- **Data versioning:** tracks the dataset with DVC.
- **Data and drift tests:** creates a reproducible, stratified 80/20 train-validation pool / held-out test split; applies controlled image corruptions to the held-out test set; compares image-feature distributions to detect input drift.
- **CI:** runs the API test suite with GitHub Actions on pushes and pull requests.

> **Important:** Input drift indicates a change in image statistics; it does not by itself prove that model accuracy has dropped. Evaluate model performance separately on labeled data.

## Architecture

```text
Oxford-IIIT Pet images
        |
        v
DVC-tracked dataset
        |
        v
Deterministic stratified split (seed 42)
        |                         |
        v                         v
Train + validation             Held-out test (20%)
        |                         |
        v                         +--> Clean reference
PyTorch training                      |
        |                              +--> Corruption suite
        v                              |    - Gaussian blur
MLflow experiments                    |    - Brightness up/down
        |                              |    - JPEG compression
        v                              |    - Downscale/upscale
Exported ONNX model                   |    - Motion blur
        |                              v
        v                         Drift report (KS distances)
FastAPI + ONNX Runtime
        |
        +--> GET /health
        +--> POST /predict
        +--> GET /metrics --> Prometheus --> Grafana dashboard
```

## Repository layout

```text
.
├── .github/workflows/main.yml     # GitHub Actions tests
├── docker-compose.monitoring.yml  # API + Prometheus + Grafana stack
├── monitoring/
│   ├── prometheus.yml             # API scrape configuration
│   └── grafana/                   # Provisioned datasource and dashboard
├── artifacts/
│   ├── model.onnx                 # Exported inference model
│   ├── classes.json               # Output-index to breed mapping
│   ├── breed_animal_types.json    # Breed-to-cat/dog mapping
│   └── models.yml                 # Training hyperparameters
├── Data.dvc                       # DVC pointer for the dataset
├── notebooks/                     # Training and ONNX export notebooks
├── reports/
│   ├── drift_baseline.json        # Clean-vs-clean threshold calibration output
│   └── drift_report.json          # Corruption drift evaluation output
├── src/final_project/
│   ├── app.py                     # FastAPI inference service
│   ├── train.py                   # PyTorch training + MLflow logging
│   ├── model.py                   # Model, training loop, evaluation metrics
│   ├── preprocessing.py           # Dataset loader
│   ├── dataset_split.py           # Reproducible stratified split
│   ├── corruption.py              # Corruption suite generation
│   ├── drift_detection.py         # Image-statistics drift analysis
│   └── calibrate_drift.py         # Exploratory drift threshold calibration
└── tests/                          # API/model, split, and drift tests
```

## Requirements

- Python **3.12**
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- Git
- Access to the dataset files and the configured DVC remote
- Optional: an NVIDIA GPU for training. Training automatically falls back to CPU if CUDA is unavailable.

The production dependency group contains the FastAPI service, ONNX Runtime, Pillow, NumPy, Uvicorn, and test/lint utilities. The development group adds PyTorch, torchvision, MLflow, DVC, OpenCV, and scikit-learn.

## Setup

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Mostafa-Ashraf-1/mlops-final_project.git
cd mlops-final_project
```

Create the project environment and install development dependencies:

```bash
uv sync --group dev
```

The project uses the `production` and `dev` dependency groups declared in `pyproject.toml`. To install only dependencies needed to run the API:

```bash
uv sync --no-dev --no-default-groups --group production --no-install-project
```

### Dataset (DVC)

The dataset is not intended to be reconstructed from the Git repository alone. `Data.dvc` tracks the data directory, and `.dvc/config` currently points to a machine-specific local path. Configure an accessible DVC remote for your environment before pulling data.

```bash
# After configuring an accessible remote:
uv run dvc pull
```

The data utilities expect the original images under:

```text
Data/oxford-iiit-pet/images/
```

Do not commit dataset images or credentials to Git.

## Run the API locally

The current checked-in ONNX model and class mappings are under `artifacts/`. Start the application from the repository root:

```bash
uv run uvicorn app:app --app-dir src/final_project --host 127.0.0.1 --port 8000
```

Open the interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Returns service status, model version, model filename, and class count. |
| `POST` | `/predict` | Accepts an uploaded image and returns a breed prediction. |
| `GET` | `/metrics` | Exposes Prometheus-formatted HTTP and prediction metrics (hidden from API docs). |

Example health check:

```bash
curl http://127.0.0.1:8000/health
```

Example prediction using an image file:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@path/to/pet.jpg"
```

A successful `/predict` response includes fields similar to:

```json
{
  "breed": "Bengal",
  "species": "cat",
  "confidence": 0.91,
  "top_3": [
    {"breed": "Bengal", "confidence": 0.91},
    {"breed": "Egyptian Mau", "confidence": 0.05},
    {"breed": "Abyssinian", "confidence": 0.02}
  ],
  "decision": "confident",
  "model_version": "v3"
}
```

The values above are an **illustrative response**, not a benchmark. The API uses a confidence threshold of `0.80`: lower top-class confidence returns `"uncertain"`. Invalid/non-image uploads are rejected, and uploads larger than 10 MB are refused.

## Run with Docker

Build the API image from the repository root:

```bash
docker build -t pets-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 pets-api
```

Then visit [http://localhost:8000/docs](http://localhost:8000/docs). The current Dockerfile installs production dependencies and copies `src/` and `artifacts/`; it does not pull the DVC dataset because the inference service does not need the training dataset.


## Monitoring (Prometheus + Grafana)

The monitoring stack observes the running inference API. It does not claim to measure model accuracy or detect semantic/data drift from live traffic; the existing offline corruption suite remains the project's input-drift evaluation.

Start the API, Prometheus, and Grafana together from the repository root:

```bash
docker compose -f docker-compose.monitoring.yml up --build -d
```

Open the services:

- **API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Prometheus:** [http://localhost:9090](http://localhost:9090)
- **Grafana dashboard:** [http://localhost:3000](http://localhost:3000) — local demo login: `admin` / `admin`

The Grafana dashboard is provisioned automatically. It displays HTTP request rate by route, average request duration, prediction decisions (confident vs. uncertain), mean prediction confidence, HTTP 5xx rate, and the prediction-confidence distribution. Prometheus scrapes `/metrics` every five seconds.

Metrics exposed by the API include:

- `pets_api_http_requests_total`: request count labelled by method, route, and status code.
- `pets_api_http_request_duration_seconds_sum` and `_count`: request duration aggregates, which can be combined in PromQL to estimate average latency.
- `pets_api_predictions_total`: successful predictions split by the API's confidence decision.
- `pets_api_prediction_confidence`: confidence histogram for successful predictions.

Send a few requests to `/predict` to populate the dashboard; metrics start at zero after a fresh API process. To stop the stack:

```bash
docker compose -f docker-compose.monitoring.yml down
```

Prometheus and Grafana data persist in named Docker volumes. The credentials above are for a local demonstration only; change them before exposing Grafana beyond your machine. This setup monitors service behavior and prediction confidence, **not ground-truth accuracy**. Confidence is not a guarantee that a prediction is correct.

## Training and experiment tracking

Training code and hyperparameters are in `src/final_project/train.py` and `artifacts/models.yml`. The experiments use PyTorch transfer learning with ResNet-18, ResNet-50, and MobileNetV3-Small.

Configure an MLflow tracking server and set `MLFLOW_URI` in your local `.env` file (see `.env.example`). Do not commit the real URI if it contains credentials.

```bash
# .env
MLFLOW_URI="http://127.0.0.1:5000"
```

Start MLflow in a separate terminal if you are running a local tracking server:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

Run the training script from the repository root after the dataset split has been generated:

```bash
uv run python -m src.final_project.train
```

Training expects images at `Data/oxford-iiit-pet/split/train/`. The training script makes a validation subset from this training directory; the held-out `split/test/` set is reserved for final evaluation and corruption/drift experiments. Hyperparameters (learning rate, epochs, and weight decay) are listed in `artifacts/models.yml`.

> Training and ONNX export are separate workflows: the API loads the already-exported `artifacts/model.onnx`. Export experiments are documented in `notebooks/export_onnx.ipynb`.

## Dataset split and corruption suite

Generate the deterministic stratified split and corruption dataset from the original images:

```bash
uv run python -m src.final_project.corruption
```

By default this creates:

```text
Data/oxford-iiit-pet/
├── images/                 # Original images; left untouched
├── split/
│   ├── train/              # 80% pool for training and validation
│   ├── test/               # Held-out 20%
│   ├── manifest.json
│   ├── train.txt
│   └── test.txt
└── corruption_test/
    ├── clean/
    ├── gaussian_blur/severity_1..3/
    ├── brightness_up/severity_1..3/
    ├── brightness_down/severity_1..3/
    ├── jpeg/severity_1..3/
    ├── downscale/severity_1..3/
    ├── motion_blur/severity_1..3/
    └── manifest.csv
```

The split uses seed `42` and retains breed representation in both sets. Corrupted variants are generated **only from the held-out test images**; the source `images/` directory is not modified. Each corruption has three severity levels.

## Drift detection

The detector extracts lightweight image statistics—brightness, contrast, sharpness, and RGB-channel means/standard deviations—and compares each corrupted scenario with the clean held-out set using the two-sample Kolmogorov–Smirnov (KS) distance.

Generate a drift report:

```bash
uv run python -m src.final_project.drift_detection
```

The default input is `Data/oxford-iiit-pet/corruption_test/`; the report is written to `reports/drift_report.json`. The default threshold is `0.10`; pass a different threshold or paths using `--help`.

Optionally estimate a threshold from repeated clean-versus-clean splits:

```bash
uv run python -m src.final_project.calibrate_drift
```

This writes `reports/drift_baseline.json` by default. It is an exploratory calibration using repeated partitions of the same clean set, not an independent estimate of production false-positive rates. Review the report and validate a threshold on separate data before treating it as an operational alert.

## Tests and CI

Run the test suite locally:

```bash
uv run pytest
```

The GitHub Actions workflow in `.github/workflows/main.yml` installs the production dependencies and runs the API/model, dataset-split, and drift-related tests on pushes and pull requests. Lint/format checks are currently present as commented-out workflow steps, so they are **not** enforced by CI.

## Reproducibility and operational notes

- The split is deterministic when the same dataset and seed (`42`) are used.
- Keep the held-out test set out of training and validation to avoid evaluation leakage.
- Keep model class order in `artifacts/classes.json` aligned with the ONNX output indices.
- The API currently uses ONNX Runtime's `CPUExecutionProvider`.
- MLflow tracking requires a reachable tracking server; set `MLFLOW_URI` before training.
- The checked-in DVC configuration uses a local Windows drive path. Replace it with a remote accessible to your machine/CI before expecting `dvc pull` to work elsewhere.
- Drift scores measure feature-distribution differences and should be interpreted alongside labeled evaluation metrics.

## Acknowledgements

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [PyTorch](https://pytorch.org/)
- [MLflow](https://mlflow.org/)
- [DVC](https://dvc.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
