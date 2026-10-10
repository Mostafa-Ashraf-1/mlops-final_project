
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image

from src.final_project.app import (
    ARTIFACTS_PATH,
    BREED_TO_SPECIES,
    CLASS_NAMES,
    MODEL_PATH,
    model,
    preprocess_image,
    get_species,
)


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_classes_file_exists():
    assert (ARTIFACTS_PATH / "classes.json").exists()


def test_breed_types_file_exists():
    assert (ARTIFACTS_PATH / "breed_animal_types.json").exists()


def test_model_is_onnx_session():
    assert isinstance(model, ort.InferenceSession)


def test_model_input_shape():
    input_meta = model.get_inputs()[0]

    assert input_meta.shape[1:] == [3, 64, 64]
    assert input_meta.type == "tensor(float)"


def test_model_output_shape():
    output_meta = model.get_outputs()[0]

    assert output_meta.shape[-1] == 37
    assert len(CLASS_NAMES) == 37


def test_class_indices_are_unique():
    with open(ARTIFACTS_PATH / "classes.json", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    indices = [int(index) for index in class_to_idx.values()]

    assert len(indices) == len(set(indices))
    assert sorted(indices) == list(range(37))


def test_every_breed_has_species():
    assert len(BREED_TO_SPECIES) >= 37

    for breed in CLASS_NAMES:
        assert get_species(breed) in {"cat", "dog"}


def test_breed_lookup_ignores_case_and_underscores():
    assert get_species("Bengal") == "cat"
    assert get_species("american_bulldog") == "dog"
    assert get_species("American Bulldog") == "dog"


def test_preprocess_image_shape_and_dtype():
    image = Image.new("RGB", (200, 150), color=(120, 80, 40))

    result = preprocess_image(image)

    assert result.shape == (1, 3, 64, 64)
    assert result.dtype == np.float32
    assert result.flags["C_CONTIGUOUS"]
    assert np.isfinite(result).all()


def test_model_inference_returns_37_scores():
    image = Image.new("RGB", (100, 100), color=(120, 150, 80))
    input_data = preprocess_image(image)

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    outputs = model.run(
        [output_name],
        {input_name: input_data},
    )

    scores = outputs[0]

    assert scores.shape == (1, 37)
    assert np.isfinite(scores).all()


def test_inference_produces_valid_top_three():
    image = Image.new("RGB", (64, 64), color=(100, 120, 140))
    input_data = preprocess_image(image)

    outputs = model.run(
        [model.get_outputs()[0].name],
        {model.get_inputs()[0].name: input_data},
    )

    scores = outputs[0][0]
    assert scores.shape == (37,)

    shifted = scores - np.max(scores)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()

    top_indices = np.argsort(probabilities)[-3:][::-1]

    assert len(top_indices) == 3
    assert len(set(top_indices.tolist())) == 3
    assert np.all((probabilities[top_indices] >= 0))
    assert np.all((probabilities[top_indices] <= 1))
    assert probabilities[top_indices[0]] >= probabilities[top_indices[1]]
    assert probabilities[top_indices[1]] >= probabilities[top_indices[2]]
