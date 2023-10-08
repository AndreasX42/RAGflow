import streamlit as st
import os
import requests
import json
from typing import Optional

API_HOST = os.environ.get("EVALBACKEND_HOST")
API_PORT = os.environ.get("EVALBACKEND_PORT")


def read_json(filename: str):
    """Load dataset from a JSON file."""
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: dict, filename: str, append: Optional[bool] = True) -> None:
    """Function used to store generated QA pairs, i.e. the ground truth."""

    # If no append should be performed we delete it
    if not append:
        os.remove(filename)

    # Check if file exists
    if os.path.exists(filename):
        # File exists, read the data
        with open(filename, "r", encoding="utf-8") as file:
            json_data = json.load(file)
            # Assuming the data is a list; you can modify as per your requirements
            json_data.extend(data)
    else:
        # File doesn't exist; set data as the new_data
        # This assumes the main structure is a list; modify as needed
        json_data = data  # [data]

    # Write the combined data back to the file
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=4)


def display_files(path: str):
    files = list_files_in_directory(path)

    if not files:
        st.write("No files uploaded!")
    else:
        # Display the files in a nice format
        st.markdown("### List of files:")
        for file_name in files:
            st.write(f"- {file_name}")

    return files


def list_files_in_directory(path):
    """List all files in the specified directory."""
    with os.scandir(path) as directory:
        return [entry.name for entry in directory if entry.is_file()]


def save_uploaded_file(uploaded_file: str, doc_save_path: str):
    """Function to save the uploaded file to the specified path."""

    if not os.path.exists(doc_save_path):
        os.makedirs(doc_save_path)

    with open(os.path.join(doc_save_path, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getvalue())


def fetch_data(endpoint):
    """Fetch data from the given endpoint."""
    response = requests.get(f"http://{API_HOST}:{API_PORT}{endpoint}")

    # Handle response errors if needed
    response.raise_for_status()

    return response.json()


def get_valid_params():
    """Fetch data for all endpoints."""
    endpoints = {
        "llm_models": "/configs/llm_models",
        "embedding_models": "/configs/embedding_models",
        "retr_search_types": "/configs/retriever_search_types",
        "grade_answer_prompts": "/configs/grade_answer_prompts",
        "grade_documents_prompts": "/configs/grade_documents_prompts",
    }

    data = {}
    for key, endpoint in endpoints.items():
        data[key] = fetch_data(endpoint)

    return data


def start_qa_gen() -> bool:
    """Start QA eval set generation."""

    json_payload = {
        "document_store_path": get_document_store_path(),
        "eval_dataset_path": get_eval_data_path(),
        "qa_gen_params_path": get_qa_gen_params_path(),
    }

    response = requests.post(
        f"http://{API_HOST}:{API_PORT}/evalsetgeneration/start", json=json_payload
    )

    # Handle response errors if needed
    response.raise_for_status()

    write_json(response.json(), get_eval_data_path(), append=False)

    return True


def get_user_directory() -> str:
    """Return the user directory based on user_id."""
    return os.path.join("./tmp", st.session_state.user_id)


def get_document_store_path() -> str:
    """Return the document store path of user."""
    return os.path.join(get_user_directory(), "document_store/")


def get_eval_data_path() -> str:
    """Return the evaluation data file path."""
    return os.path.join(get_user_directory(), "eval_data.json")


def get_eval_params_path() -> str:
    """Return the evaluation hyperparameters file path."""
    return os.path.join(get_user_directory(), "eval_params.json")


def get_eval_results_path() -> str:
    """Return the evaluation results of provided hyperparameters."""
    return os.path.join(get_user_directory(), "eval_results.json")


def get_hp_runs_data_path() -> str:
    """Return the generated data of the hyperparameter runs."""
    return os.path.join(get_user_directory(), "hp_runs_data.csv")


def get_qa_gen_params_path() -> str:
    """Return the parameters for the eval set generation."""
    return os.path.join(get_user_directory(), "qa_gen_params.json")
