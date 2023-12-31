import os
from typing import Optional
import requests
import pytest

# Constants
RAGFLOW_HOST = os.environ.get("RAGFLOW_HOST")
RAGFLOW_PORT = os.environ.get("RAGFLOW_PORT")

CHROMADB_HOST = os.environ.get("CHROMADB_HOST")
CHROMADB_PORT = os.environ.get("CHROMADB_PORT")

LABEL_DATASET_ENDPOINT = "/generation"
HP_EVALUATION_ENDPOINT = "/evaluation"

DOCUMENT_STORE_PATH = "./resources/document_store"
LABEL_DATASET_PATH = "./resources/input_label_dataset.json"
HYPERPARAMETERS_PATH = "./resources/input_hyperparameters.json"
HYPERPARAMETERS_RESULTS_PATH = "./resources/output_hyperparameters_results.json"
HYPERPARAMETERS_RESULTS_DATA_PATH = (
    "./resources/output_hyperparameters_results_data.csv"
)
# pytest fixtures


@pytest.fixture
def user_id_without_upsert() -> str:
    """Returns a test user id of 0."""
    return "1"


@pytest.fixture
def first_user_id() -> str:
    """Returns a test user id of 0."""
    return "2"


@pytest.fixture
def second_user_id() -> str:
    """Returns a test user id of 1."""
    return "3"


# helper functions
def fetch_data(
    method: str,
    host: str,
    port: str,
    endpoint: str,
    payload: Optional[dict] = None,
):
    """Fetch data from the given endpoint."""

    if method == "get":
        response = requests.get(f"http://{host}:{port}{endpoint}")
    else:
        response = requests.post(f"http://{host}:{port}{endpoint}", json=payload)

    # Handle response errors if needed
    response.raise_for_status()

    return response
