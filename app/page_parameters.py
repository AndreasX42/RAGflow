import streamlit as st
from utils import *
from utils import display_user_login_warning


def page_parameters():
    st.title("Parameters Page")
    st.subheader("Provide parameters for building and evaluating the system.")

    if display_user_login_warning():
        return

    tab1, tab2, tab3 = st.tabs(
        ["QA Generator settings", "Hyperparameters settings", "ReadMe"]
    )

    valid_data = get_valid_params()

    with tab1:
        provide_qa_gen_form(valid_data)

        upload_files(
            context="qa_params",
            dropdown_msg="Upload JSON file",
            ext_list=["json"],
            file_path=get_qa_gen_params_path(),
        )

        st.markdown("<br>" * 1, unsafe_allow_html=True)

        submit_button = st.button("Start eval set generator", key="SubmitQA")

        if submit_button:
            with st.spinner("Running eval set generator..."):
                result = start_qa_gen()

                if "Success" in result:
                    st.success(result)
                else:
                    st.error(result)

    with tab2:
        provide_hp_params_form(valid_data)

        upload_files(
            context="hp_params",
            dropdown_msg="Upload JSON file",
            ext_list=["json"],
            file_path=get_eval_params_path(),
        )

        st.markdown("<br>" * 1, unsafe_allow_html=True)

        submit_button = st.button("Start hyperparameter evaluation", key="SubmitHP")

        if submit_button:
            with st.spinner("Running hyperparameter evaluation..."):
                result = start_hp_run()

                if "Success" in result:
                    st.success(result)
                else:
                    st.error(result)

    with tab3:
        st.subheader(
            "The app and the backend expect a handful of different files with the correct names in order to work properly. Below are the required files and their descriptions."
        )

        # Helper function to display paths
        def display_path(description, path_func):
            st.markdown(
                f"**:file_folder: {description}**\n> {path_func()}\n---",
                unsafe_allow_html=True,
            )

        # Display each path with the helper function
        display_path(
            "User's directory, consists of an UUID linked to the user.",
            get_user_directory,
        )
        display_path("The Document Store folder:", get_document_store_path)
        display_path(
            "JSON file with parameters to generate the question-context-answer triples. The resulting evaluation dataset is written to the next file below.",
            get_qa_gen_params_path,
        )
        display_path(
            "JSON file with the generated evaluation dataset used for benchmarking RAG systems:",
            get_eval_data_path,
        )
        display_path(
            "JSON file with hyperparameters used to build the corresponding RAG application:",
            get_eval_params_path,
        )
        display_path(
            "JSON file with benchmarks/metrics of the RAG system built with provided hyperparameters:",
            get_eval_results_path,
        )
        display_path(
            "CSV file with additional data from each hyperparameter evaluation run, including predicted answers and corresponding document chunks:",
            get_hp_runs_data_path,
        )


def provide_hp_params_form(valid_data: dict):
    with st.expander("Hyperparameters settings"):
        attributes2 = {
            "id": st.number_input("ID", value=0, key="id for this HP config"),
            "chunk_size": st.number_input("Chunk Size", value=512, key="chunk size"),
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
            "similarity_method": st.selectbox(
                "Retriever Similarity Method", valid_data["retr_sim_method"]
            ),
            "search_type": st.selectbox(
                "Retriever Search Type", valid_data["retr_search_types"]
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


def provide_qa_gen_form(valid_data: dict):
    with st.expander("Provide QA Generator settings form"):
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

        persist_to_vs = st.checkbox(
            "Cache answer embeddings in ChromaDB for each of the models listed below",
            value=False,
            key="persist_to_vs",
        )

        # Get input from the user
        if persist_to_vs:
            input_text = st.text_area(
                "Enter a list of embedding model names for caching the generated answer embeddings in chromadb (one per line):"
            )

        submit_button = st.button("Submit", key="submit1")
        if submit_button:
            # save bool in dict
            attributes["persist_to_vs"] = persist_to_vs

            if persist_to_vs:
                # Store the list in a dictionary
                # Split the text by newline to get a list
                text_list = input_text.split("\n")
                attributes["embedding_model_list"] = text_list
            else:
                attributes["embedding_model_list"] = []

            # flag not necessary for user
            attributes["generate_eval_set"] = True

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
