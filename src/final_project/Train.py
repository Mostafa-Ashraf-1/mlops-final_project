import os
import time
from pathlib import Path

import mlflow
import torch
import torchvision.transforms as T
import yaml
from dotenv import load_dotenv
from src.final_project import convertCuda
from src.final_project.convertCuda import DeviceDataLoader
from src.final_project.model import PetsModel, fit
from src.final_project.Preprocessing import PetsDataset
from torch.utils.data import DataLoader, random_split
from torchvision import models
from mlflow.models import infer_signature

load_dotenv()
MLFLOW_URI = os.getenv("MLFLOW_URI")
REGISTERED_MODEL_NAME = "PetsModel"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("pets-model")

BASE_PATH = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_PATH = BASE_PATH / "artifacts"
DATA_PATH = BASE_PATH / "Data" / "oxford-iiit-pet"
SPLIT_PATH = DATA_PATH  / "split"
TRAIN_IMAGES_PATH = SPLIT_PATH / "train"

IMG_SIZE = (224, 224)
VAL_PCT = 0.1
BATCH_SIZE = 128
ImageNet_STATE = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

resnet18 = "ResNet-18"
resnet50 = "ResNet-50"
mobilenet_small = "MobileNetV3-Small"


transformations = T.Compose([
                      T.Resize(IMG_SIZE),
                      T.ToTensor(),
                      T.Pad(8, padding_mode='reflect'),
                      T.RandomCrop(IMG_SIZE),
                      T.Normalize(*ImageNet_STATE),
                      T.RandomHorizontalFlip(0.3),
                      T.GaussianBlur(3)
                      ]
                      )

dataset = PetsDataset(TRAIN_IMAGES_PATH, transformations)

NUM_CLASSES = len(dataset.classes)

val_size = int(VAL_PCT * len(dataset))
train_size = len(dataset) - val_size

train_ds, valid_ds = random_split(dataset, [train_size, val_size])

train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
valid_dl = DataLoader(valid_ds, BATCH_SIZE * 2, num_workers=0, pin_memory=True)

device = convertCuda.get_default_device()
trainloader = DeviceDataLoader(train_dl, device)
valloader = DeviceDataLoader(valid_dl, device)


with open(f'{ARTIFACTS_PATH}/models.yml') as f:
    classifiers = yaml.safe_load(f)['models']


MODEL_ZOO = [
    (resnet18, classifiers[resnet18]['params'], models.resnet18),
    (resnet50, classifiers[resnet50]['params'], models.resnet50),
    (mobilenet_small, classifiers[mobilenet_small]['params'], models.mobilenet_v3_small)
]

for name, params, classifier in MODEL_ZOO:

    model = PetsModel(NUM_CLASSES, classifier, pretrained=True)
    convertCuda.to_device(model, device)

    with mlflow.start_run(run_name=name) as run:

        t0 = time.perf_counter()

        # mlflow.log_params(Base_params)

        history = fit(
            epochs=params['epochs'],
            lr=params['lr'],
            model=model,
            train_loader=trainloader,
            val_loader=valloader,
            opt_func=torch.optim.Adam,
            weight_decay=params['weight_decay']
        )

        train_seconds = time.perf_counter() - t0
        train_minutes = train_seconds / 60

        mlflow.log_metric("train_minutes", train_minutes)
        mlflow.log_metric("epochs", params['epochs'])
        model.eval()

        example_input = torch.randn(
            1, 3, 224, 224,
            device=device
        )

        with torch.no_grad():
            example_output = model(example_input)

        signature = infer_signature(
            example_input.cpu().numpy(),
            example_output.cpu().numpy()
        )

        mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
            input_example=example_input.cpu().numpy(),
            signature=signature,
        )