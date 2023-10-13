import streamlit as st
import os
import requests
import json
from typing import Optional
import time

API_HOST = os.environ.get("EVALBACKEND_HOST")
API_PORT = os.environ.get("EVALBACKEND_PORT")


def read_json(filename: str):
    """Load dataset from a JSON file."""
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: dict, filename: str, append: Optional[bool] = True) -> None:
    """Function used to store generated QA pairs, i.e. the ground truth."""

    # If no append should be performed we delete it
    if not append and os.path.exists(filename):
        os.remove(filename)

    combined_data = []

    # Check if file exists
    if os.path.exists(filename):
        # File exists, read the data
        with open(filename, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
            if isinstance(existing_data, list):
                combined_data.extend(existing_data)
            else:
                combined_data.append(existing_data)

    # Append the new data
    if isinstance(data, list):
        combined_data.extend(data)
    else:
        combined_data.append(data)

    # Write the combined data back to the file
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(combined_data, file, indent=4)


def list_files_in_directory(path):
    """List all files in the specified directory and its subdirectories."""
    file_list = []

    for root, dirs, files in os.walk(path):
        for file in files:
            # Create a relative path from the original path
            relative_path = os.path.relpath(root, path)
            if relative_path == ".":
                # If the file is in the root directory, just append the file name
                file_list.append(file)
            else:
                # Otherwise, append the subdirectory prefix and the file name
                file_list.append(os.path.join(relative_path, file))

    return file_list


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

    try:
        json_payload = {
            "document_store_path": get_document_store_path(),
            "eval_dataset_path": get_eval_data_path(),
            "qa_gen_params_path": get_qa_gen_params_path(),
            "user_id": st.session_state.user_id,
        }

        response = requests.post(
            f"http://{API_HOST}:{API_PORT}/evalsetgeneration/start", json=json_payload
        )

        response.raise_for_status()
        return "Successfully generated evaluation data."

    except requests.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"

    except Exception as err:
        return f"An error occurred: {err}"


def start_hp_run() -> bool:
    """Start Hyperparameters evaluation."""

    try:
        json_payload = {
            "document_store_path": get_document_store_path(),
            "eval_dataset_path": get_eval_data_path(),
            "eval_params_path": get_eval_params_path(),
            "eval_results_path": get_eval_results_path(),
            "hp_runs_data_path": get_hp_runs_data_path(),
            "user_id": st.session_state.user_id,
        }

        response = requests.post(
            f"http://{API_HOST}:{API_PORT}/evaluations/start", json=json_payload
        )

        response.raise_for_status()
        return "Successfully ran hyperparameters evaluations."

    except requests.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"

    except Exception as err:
        return f"An error occurred: {err}"


def upload_files(
    context: str,
    ext_list: list[str],
    file_path: str,
    allow_multiple_files: Optional[bool] = False,
):
    """Function to let the user upload files."""
    with st.expander("Upload file(s)"):
        with st.form(f"upload_{context}", clear_on_submit=True):
            selection = st.file_uploader(
                label=f"Upload file(s) with extension {', '.join(ext_list)}.",
                type=ext_list,
                accept_multiple_files=allow_multiple_files,
                key=f"context_{context}",
            )

            submitted = st.form_submit_button("Upload file(s)")

            # for now only ["json"] or ["docx", "txt", "pdf"] possible as ext_list
            if submitted and selection is not None:
                if "json" in ext_list:
                    # To read file as bytes:
                    bytes_data = selection.getvalue()
                    # To convert to a string based on json:
                    string_data = bytes_data.decode("utf-8")
                    # Load the JSON to a Python object
                    data = json.loads(string_data)
                    # Display the JSON contents
                    st.write(data)
                    # save json
                    write_json(
                        data,
                        file_path,
                    )

                    with st.spinner():
                        st.success("File(s) uploaded successfully!")
                        time.sleep(2)

                else:
                    for file in selection:
                        save_uploaded_file(file, file_path)

                    with st.spinner():
                        st.success("File(s) uploaded successfully!")
                        time.sleep(2)


def realname(path, root=None):
    """Helper function for ptree."""
    if root is not None:
        path = os.path.join(root, path)
    result = os.path.basename(path)
    if os.path.islink(path):
        realpath = os.readlink(path)
        result = "%s -> %s" % (os.path.basename(path), realpath)
    return result


def ptree(startpath, depth=-1):
    """Displays a tree structure of current user directory."""
    prefix = 0
    if startpath != "/":
        if startpath.endswith("/"):
            startpath = startpath[:-1]
        prefix = len(startpath)

    structure_str = ""
    for root, dirs, files in os.walk(startpath):
        level = root[prefix:].count(os.sep)
        if depth > -1 and level > depth:
            continue
        indent = subindent = ""
        if level > 0:
            indent = "|   " * (level - 1) + "|-- "
        subindent = "|   " * (level) + "|-- "
        structure_str += "{}{}/\n".format(indent, realname(root))
        for d in dirs:
            if os.path.islink(os.path.join(root, d)):
                structure_str += "{}{}\n".format(subindent, realname(d, root=root))
        for f in files:
            structure_str += "{}{}\n".format(subindent, realname(f, root=root))
    return structure_str


#
# Getter methods for user file paths.
#
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
