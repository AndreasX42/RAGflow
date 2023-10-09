import streamlit as st
import json
from utils import *


def page_params():
    st.title("Parameters Page")
    st.subheader("Provide parameters for building and evaluating the system.")

    uploaded_file = None

    if "user_id" not in st.session_state or not st.session_state.user_id:
        st.warning("WARNING: Authenticate before uploading data.")

    if "file_uploaded" not in st.session_state or uploaded_file is not None:
        st.session_state.file_uploaded = False
        uploaded_file = None

    tab1, tab2 = st.tabs(["QA Generator settings", "Hyperparameters settings"])

    valid_data = get_valid_params()

    with tab1:
        if "qa_gen_running" not in st.session_state:
            st.session_state.qa_gen_running = False
            st.session_state.response = 0

        submit_button = st.button("Start eval set generator", key="Submit3")
        if submit_button:
            st.session_state.qa_gen_running = True
            start_qa_gen()

        if st.session_state.response > 0:
            if st.session_state.response == 1:
                st.success("QA Eval Set Generation Completed!")
            if st.session_state.response == 2:
                st.error("Error occurred during QA Eval Set Generation.")
            st.session_state.qa_gen_running = False
            st.session_state.response = 0

        if st.session_state.qa_gen_running:
            with st.spinner("Running generator..."):
                st.code("waiting...")

        else:
            st.markdown("<br>" * 1, unsafe_allow_html=True)
            with st.expander("QA Generator settings"):
                attributes = {
                    "chunk_size": st.number_input("Chunk Size", value=512),
                    "chunk_overlap": st.number_input("Chunk Overlap", value=10),
                    "length_function_name": st.selectbox(
                        "Length Function for splitting or embedding model for corresponding tokenizer",
                        valid_data["embedding_models"] + ["len"],
                    ),
                    "qa_generator_llm": st.selectbox(
                        "QA Language Model", valid_data["llm_models"]
                    ),
                }
                # Get input from the user
                input_text = st.text_area(
                    "Enter a list of embedding model names for caching the generated answer embeddings in chromadb (one per line):"
                )
                # Split the text by newline to get a list
                text_list = input_text.split("\n")
                # Store the list in a dictionary
                attributes["embedding_models"] = text_list
                submit_button = st.button("Submit", key="submit1")
                if submit_button:
                    with st.spinner("Saving to file."):
                        st.write("Saved to file. You've entered the following values:")
                        # This is just a mockup to show you how to display the attributes.
                        # You'll probably want to process or display them differently.
                        st.write(attributes)
                        # save json
                        write_json(
                            attributes,
                            get_qa_gen_params_path(),
                        )

            st.markdown("<br>" * 1, unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                label="OR: Upload a JSON file to provide a list of hyperparameters instead.",
                type=["json"],
                key="qa_json",
            )

            if uploaded_file is not None:
                st.session_state.file_uploaded = True

                # To read file as bytes:
                bytes_data = uploaded_file.getvalue()

                # To convert to a string based on json:
                string_data = bytes_data.decode("utf-8")

                # Load the JSON to a Python object
                data = json.loads(string_data)

                # Display the JSON contents
                st.write(data)

                # Your dictionary
                st.text("Example of expected input:")

                # save json
                write_json(
                    data,
                    get_qa_gen_params_path(),
                )

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

    with tab2:
        with st.expander("Hyperparameters settings"):
            attributes2 = {
                "id": st.number_input("ID", value=0, key="id for this HP config"),
                "chunk_size": st.number_input(
                    "Chunk Size", value=512, key="chunk size"
                ),
                "chunk_overlap": st.number_input(
                    "Chunk Overlap", value=10, key="chunk overlap"
                ),
                "length_function_name": st.selectbox(
                    "Length Function for splitting or embedding model for corresponding tokenizer",
                    valid_data["embedding_models"] + ["len"],
                    key="len_tab2",
                ),
                "num_retrieved_docs": st.number_input(
                    "Number of Docs the retriever should return",
                    value=3,
                    key="number of docs to retrieve",
                ),
                "search_type": st.selectbox(
                    "Search Type", valid_data["retr_search_types"]
                ),
                "embedding_model": st.selectbox(
                    "Embedding Model",
                    valid_data["embedding_models"],
                    key="Name of embedding model.",
                ),
                "qa_llm": st.selectbox(
                    "QA Language Model",
                    valid_data["llm_models"],
                    key="Name of LLM for QA task.",
                ),
            }

            use_llm_grader = st.checkbox(
                "Use LLM Grader", value=False, key="use_llm_grader_checkbox"
            )

            if use_llm_grader:
                attributes2["grade_answer_prompt"] = st.selectbox(
                    "Grade Answer Prompt",
                    valid_data["grade_answer_prompts"],
                    key="Type of prompt to use for answer grading.",
                )
                attributes2["grade_docs_prompt"] = st.selectbox(
                    "Grade Documents Prompt",
                    valid_data["grade_documents_prompts"],
                    key="Type of prompt to grade retrieved document chunks.",
                )
                attributes2["grader_llm"] = st.selectbox(
                    "Grading LLM",
                    valid_data["llm_models"],
                    key="Name of LLM for grading.",
                )

            submit_button = st.button("Submit", key="Submit2")
            if submit_button:
                attributes2["use_llm_grader"] = use_llm_grader
                st.write("Saved to file. You've entered the following values:")
                # This is just a mockup to show you how to display the attributes.
                # You'll probably want to process or display them differently.
                st.write(attributes2)
                write_json(
                    attributes2,
                    get_eval_params_path(),
                    append=True,
                )

        st.markdown("<br>" * 1, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            label="OR: Upload a JSON file to provide a list of hyperparameters instead.",
            type=["json"],
            key="hp_json",
        )

        if uploaded_file is not None:
            st.session_state.file_uploaded = True

            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()

            # To convert to a string based on json:
            string_data = bytes_data.decode("utf-8")

            # Load the JSON to a Python object
            data = json.loads(string_data)

            # Display the JSON contents
            st.write(data)
            write_json(
                data,
                get_eval_params_path(),
                append=True,
            )

        # Your dictionary
        st.text("Example of expected input:")

        st.code(
            """
        {
            "chunk_size": 1024,
            "chunk_overlap": 10,
            "num_retrieved_docs": 3,
            "length_function_name": "len",
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
