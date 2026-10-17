# Pet Breed Classification — MLOps Pipeline

An end-to-end MLOps project that classifies images into **37 cat and dog breeds** from the Oxford-IIIT Pet dataset. It combines PyTorch training and MLflow experiment tracking, DVC-based data versioning, ONNX Runtime inference through FastAPI, automated tests, corruption-based input-drift evaluation, and Prometheus/Grafana service monitoring.

## Quick start: get a prediction

Requirements: Docker and a local pet image such as `pet.jpg`.

Run these commands from the repository root:

```bash
docker build -t pets-api .
docker run --rm -d --name pets-api -p 8000:8000 pets-api
curl -X POST "http://localhost:8000/predict" -F "file=@pet.jpg"
```

Replace `pet.jpg` with the path to an image on your computer. The response includes the predicted breed and species, confidence, top-three predictions, a `confident` / `uncertain` decision, and model version. The confidence threshold is `0.80`; confidence is not a guarantee of correctness.

- API health: [http://localhost:8000/health](http://localhost:8000/health)
- Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Stop the standalone API container when finished:

```bash
docker stop pets-api
```

**Important:** stop this standalone container before starting the monitoring Compose stack below. Both setups use host port 8000 and the container name `pets-api`.

## What the project includes

- **Classification:** PyTorch transfer-learning experiments with ResNet-18, ResNet-50, and MobileNetV3-Small.
- **Experiment tracking:** MLflow logs training/validation metrics including loss, top-1 accuracy, macro-F1, coverage, selective accuracy, and expected calibration error (ECE).
- **Inference:** the checked-in `artifacts/model.onnx` is served by FastAPI using ONNX Runtime's CPU execution provider.
- **Data versioning:** DVC tracks the dataset; remote availability depends on your DVC configuration.
- **Input-drift evaluation:** a stratified held-out test set is used to generate blur, brightness, JPEG, downscale/upscale, and motion-blur variants at three severity levels. Image statistics are compared with two-sample Kolmogorov–Smirnov (KS) distances.
- **Service monitoring:** Prometheus scrapes API metrics and Grafana displays request rate, latency, errors, and prediction-confidence/decision metrics.
- **Automation:** GitHub Actions runs the test suite on pushes and pull requests.

Input drift is evidence of a change in image-feature distributions; it does **not** by itself prove model accuracy has fallen. The live monitoring dashboard tracks service behavior and prediction confidence, not ground-truth accuracy.

## Architecture

```text
Oxford-IIIT Pet images
        |
        v
DVC-managed data (when remote is configured)
        |
        v
Deterministic stratified 80/20 split (seed 42)
        |                              |
        v                              v
Train/validation pool             Held-out test (20%)
        |                              |
        v                              +--> Clean reference
PyTorch training                    +--> Corruption suite
        |                              |
        v                              v
MLflow experiments               KS-based drift report
        |
        v
Exported ONNX model
        |
        v
FastAPI + ONNX Runtime (CPU)
   |            |
   |            +--> /metrics --> Prometheus --> Grafana
   +--> /health and /predict
```

## Repository layout

```text
.
├── .github/workflows/main.yml
├── artifacts/
│   ├── model.onnx
│   ├── classes.json
│   ├── breed_animal_types.json
│   └── models.yml
├── Data.dvc
├── docker-compose.monitoring.yml
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── provisioning/
│       └── dashboards/
├── notebooks/                     # Training and ONNX export notebooks
├── reports/                       # Drift report and exploratory baseline
├── src/final_project/
│   ├── app.py
│   ├── train.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── dataset_split.py
│   ├── corruption.py
│   ├── drift_detection.py
│   └── calibrate_drift.py
└── tests/
```

## Requirements and environment setup

- Python **3.12**
- [uv](https://docs.astral.sh/uv/)
- Git
- Docker Engine/Desktop with the Compose plugin (for containerized serving and monitoring)
- Dataset images for data preparation and training

Clone the repository:

```bash
git clone https://github.com/Mostafa-Ashraf-1/mlops-final_project.git
cd mlops-final_project
```

Install **both** dependency groups from the repository root. The `production` group provides the API dependencies; the `dev` group adds training, MLflow, DVC, OpenCV, and scikit-learn dependencies.

```bash
uv sync --group production --group dev
```

For the API container, the Dockerfile installs only the production group automatically; you do not need to install the training dependencies to serve the checked-in ONNX model.

## Dataset setup (DVC)

The data-preparation scripts expect original image files directly under:

```text
Data/oxford-iiit-pet/images/
```

The repository's checked-in `.dvc/config` points to a **machine-specific Windows local path**. It will not work on another computer unless that path exists there. Before running `dvc pull`, configure a DVC remote that is accessible from your machine and contains the tracked data:

```bash
uv run dvc remote list
uv run dvc pull
```

If the configured remote is unavailable, set up an accessible remote with your dataset or place the original dataset images in the expected directory yourself. A successful Git clone alone does not guarantee the dataset is available. Do not commit dataset images or credentials to Git.

## Prepare the dataset split and corruption suite

From the repository root, run:

```bash
uv run python -m src.final_project.corruption
```

This command **creates/rebuilds the split first**, then creates clean and corrupted copies from the held-out test images. The original `images/` directory is left untouched. The split generator rebuilds `Data/oxford-iiit-pet/split/` on each run, so do not keep unique files inside that generated directory.

The output structure is:

```text
Data/oxford-iiit-pet/
├── images/                 # Original images; not modified
├── split/
│   ├── train/              # 80% pool used for training/validation
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

The split uses seed `42` and stratifies by breed. The held-out test set must remain excluded from training and validation. The training script creates a validation subset from `split/train/`.

If you only need the split (without generating the corruptions), run:

```bash
uv run python -m src.final_project.dataset_split
```

## Train models and track experiments with MLflow

Training expects the split to exist and reads images from `Data/oxford-iiit-pet/split/train/`. It runs ResNet-18, ResNet-50, and MobileNetV3-Small experiments using the settings in `artifacts/models.yml`.

1. Create a local `.env` from `.env.example` and set the tracking URI. For a local tracking server, use:

   ```dotenv
   MLFLOW_URI="http://127.0.0.1:5000"
   ```

2. In a separate terminal, start the local MLflow server:

   ```bash
   uv run mlflow server --host 127.0.0.1 --port 5000
   ```

3. From the repository root, run training:

   ```bash
   uv run python -m src.final_project.train
   ```

Training and ONNX export are separate workflows. The inference API loads the existing `artifacts/model.onnx`; it does not train or export a model at startup. Export experiments are documented in `notebooks/export_onnx.ipynb`.

Do not commit secrets or private tracking URLs in `.env`.

## Run the API locally

After installing both dependency groups, run the service from the repository root:

```bash
uv run uvicorn app:app --app-dir src/final_project --host 127.0.0.1 --port 8000
```

Endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service status, model version/name, and class count |
| `POST` | `/predict` | Image upload and breed prediction |
| `GET` | `/metrics` | Prometheus-formatted HTTP and prediction metrics; hidden from the interactive docs |

Example prediction:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@path/to/pet.jpg"
```

The upload field must be named `file`. The API rejects non-image content types, empty/invalid images, and uploads larger than 10 MB.

## Run the API with Docker

Build and start the inference service:

```bash
docker build -t pets-api .
docker run --rm -d --name pets-api -p 8000:8000 pets-api
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs). The image includes `src/` and `artifacts/` and does not need the training dataset or DVC to serve predictions.

Stop it with:

```bash
docker stop pets-api
```

## Monitoring with Prometheus and Grafana

The Compose file starts the API, Prometheus, and Grafana together. **If the standalone `pets-api` container is already running, stop it first** to free port 8000 and the container name.

Start the full stack from the repository root:

```bash
docker compose -f docker-compose.monitoring.yml up --build -d
```

Open:

- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Prometheus targets: [http://localhost:9090/targets](http://localhost:9090/targets)
- Grafana: [http://localhost:3000](http://localhost:3000)

The local demo Grafana login is `admin` / `admin`. Change these credentials before exposing Grafana outside your machine. Prometheus should show the `pets-api` target as **UP**. It scrapes the API's `/metrics` endpoint every five seconds through the Compose service name.

The provisioned Grafana dashboard shows:

- HTTP request rate by route
- Average request duration
- Confident versus uncertain prediction counts/rates
- Mean prediction confidence and confidence distribution
- HTTP 5xx response rate

The API metrics include `pets_api_http_requests_total`, `pets_api_http_request_duration_seconds_sum`, `pets_api_http_request_duration_seconds_count`, `pets_api_predictions_total`, and the `pets_api_prediction_confidence` histogram. Make several `/predict` requests to populate prediction panels. Counters are held in API process memory and reset when the API process restarts; Prometheus/Grafana data use named Docker volumes.

Stop the stack:

```bash
docker compose -f docker-compose.monitoring.yml down
```

This dashboard monitors service behavior and prediction confidence. It does not measure ground-truth accuracy or replace the offline corruption-based drift report.

## Run the offline drift detector

After generating the split and corruption suite, run:

```bash
uv run python -m src.final_project.drift_detection
```

By default, the detector compares image-feature distributions from each corruption/severity directory with the clean held-out images in `Data/oxford-iiit-pet/split/test/`. It extracts brightness, contrast, sharpness, and RGB-channel means/standard deviations, then computes a two-sample KS distance per feature. A scenario is flagged when any feature's distance exceeds the default threshold `0.10`.

The JSON report is written to `reports/drift_report.json`. Override the inputs, threshold, or output path with command-line arguments:

```bash
uv run python -m src.final_project.drift_detection --help
```

Optional exploratory threshold calibration:

```bash
uv run python -m src.final_project.calibrate_drift
```

This writes `reports/drift_baseline.json` by default. It estimates natural clean-versus-clean variation by repeatedly splitting the same clean test set; it is **not** an independent estimate of production false-positive rates. Validate a threshold on separate data before using it as an operational alert.

## Tests and CI

Install both dependency groups, then run from the repository root:

```bash
PYTHONPATH=src uv run pytest
```

The `PYTHONPATH=src` setting allows tests that import the `final_project` package to resolve the source tree. GitHub Actions runs tests on pushes and pull requests. Ruff lint/format checks are present as commented-out steps in the current workflow and therefore are not enforced by CI.

## Troubleshooting

- **`dvc pull` fails:** check `uv run dvc remote list` and configure a remote that is reachable from your machine. The committed local Windows path is not portable.
- **Training cannot connect to MLflow:** ensure the MLflow server is running and `MLFLOW_URI` in `.env` matches its address.
- **Prometheus target is DOWN:** confirm the Compose stack is running with `docker compose -f docker-compose.monitoring.yml ps`; check [http://localhost:9090/targets](http://localhost:9090/targets) and inspect logs with `docker compose -f docker-compose.monitoring.yml logs --tail=80`.
- **Port 8000 is already allocated:** stop the standalone API container before launching the monitoring Compose stack.
- **Grafana panels are empty:** send several requests to `POST /predict`, then wait for one or two Prometheus scrapes. Check that the `pets-api` target is UP.
- **Drift report finds no scenarios:** rerun `uv run python -m src.final_project.corruption` to rebuild the split and corruption suite.

## Reproducibility notes

- The split is deterministic for the same input files, test size, and seed (`42`).
- The original `images/` directory is preserved; generated `split/` and corruption outputs can be rebuilt.
- Keep the held-out test set out of training and validation.
- Keep `artifacts/classes.json` aligned with ONNX output indices.
- The API currently uses ONNX Runtime's `CPUExecutionProvider`.
- Drift scores are feature-distribution signals and should be interpreted alongside evaluation on labeled data.

## Acknowledgements

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [PyTorch](https://pytorch.org/)
- [MLflow](https://mlflow.org/)
- [DVC](https://dvc.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)
