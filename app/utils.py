import streamlit as st
import os
import requests
import json
from typing import Optional
import time

API_HOST = os.environ.get("EVALBACKEND_HOST")
API_PORT = os.environ.get("EVALBACKEND_PORT")

MAX_FILE_SIZE = 10 * 1024**2  # 10MB according to .streamlit/config


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
        "retr_sim_method": "/configs/retriever_similarity_methods",
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
            "api_keys": st.session_state.api_keys,
        }

        response = requests.post(
            f"http://{API_HOST}:{API_PORT}/evalsetgeneration/start", json=json_payload
        )

        response.raise_for_status()
        return "Successfully generated evaluation data."

    except requests.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}\n\nMessage: {response.text}"

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
            "api_keys": st.session_state.api_keys,
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
    dropdown_msg: str,
    ext_list: list[str],
    file_path: str,
    allow_multiple_files: Optional[bool] = False,
):
    """Function to let the user upload files."""

    with st.expander(dropdown_msg):
        with st.form(f"upload_{context}", clear_on_submit=True):
            selection = st.file_uploader(
                label=f"Upload file(s) of format extension {', '.join(ext_list)}.",
                type=ext_list,
                accept_multiple_files=allow_multiple_files,
                key=f"context_{context}",
            )

            submitted = st.form_submit_button("Upload file(s)")

            # for now only ["json"] or ["docx", "txt", "pdf"] possible as ext_list
            if submitted and selection is not None:
                with st.spinner():
                    if ext_list == ["json"]:
                        success = process_json_file(selection, file_path)
                        if success:
                            st.success("File(s) uploaded successfully!")
                            time.sleep(1)
                    else:
                        success = process_other_files(selection, file_path, ext_list)
                        if success:
                            st.success("File(s) uploaded successfully!")
                            time.sleep(1)

        if context == "qa_params":
            st.markdown("<br>" * 1, unsafe_allow_html=True)
            st.text("Example of expected JSON input:")
            st.code(
                """
            {
                "chunk_size": 2048,
                "chunk_overlap": 0,
                "length_function_name": "text-embedding-ada-002",
                "qa_generator_llm": "gpt-3.5-turbo",
                "generate_eval_set": true, # to generate evaluation set, if false we use we load existing set
                "persist_to_vs": true # if true, for now chromadb resets all user collections
                "embedding_model_list": list[embedding model names] # list of embedding model names to use for caching in chromadb
            }
            
            """
            )

        if context == "hp_params":
            st.markdown("<br>" * 1, unsafe_allow_html=True)
            st.text("Example of expected JSON input:")
            st.code(
                """
            {
                "chunk_size": 1024,
                "chunk_overlap": 10,
                "num_retrieved_docs": 3,
                "length_function_name": "len",
                "similarity_method": "cosine",
                "search_type": "mmr",
                "embedding_model": "text-embedding-ada-002",
                "qa_llm": "gpt-3.5-turbo",
                "use_llm_grader": true,
                "grade_answer_prompt": "few_shot",
                "grade_docs_prompt": "default",
                "grader_llm": "gpt-3.5-turbo",
            }
            
            # if use_llm_grader=False, no additional parameters have to be declared
            """
            )


def process_json_file(file, file_path):
    # Ensure file size is within limits
    if file.size > MAX_FILE_SIZE:
        st.error(f"File size exceeds {MAX_FILE_SIZE} MB!")
        return False

    # Attempt to parse JSON to ensure validity
    bytes_data = file.getvalue()
    try:
        string_data = bytes_data.decode("utf-8")
        data = json.loads(string_data)

    except Exception:
        st.error("Invalid file format, expected JSON file!")
        return False

    # Display the JSON contents and save
    st.write(data)
    write_json(data, file_path)

    return True


def process_other_files(selection, file_path, valid_formats):
    for file in selection:
        # Ensure file size is within limits
        if file.size > MAX_FILE_SIZE:
            st.error(f"File '{file.name}' exceeds {MAX_FILE_SIZE} MB!")
            return False

        # For additional validations on file content, insert here
        _, extension = os.path.splitext(file.name)
        if extension[1:] not in valid_formats:
            st.error(f"File '{file.name}' has invalid format!")
            return False

        # Save the file
        save_uploaded_file(file, file_path)

    return True


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


def display_user_login_warning():
    """Default function to warn users to login before using service."""
    if "user_id" not in st.session_state or not st.session_state.user_id:
        st.warning("Warning: You need to login before being able to use this service.")
        return True


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
