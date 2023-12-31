import json
import pandas as pd
import streamlit as st
from utils import *

from streamlit import chat_message


def page_parameters():
    st.title("Parameters Page")
    st.subheader("Provide parameters for building and evaluating the system.")

    if display_user_login_warning():
        return

    tab1, tab2 = st.tabs(["QA Generator settings", "Hyperparameters settings"])

    valid_data = get_valid_params()

    with tab1:
        st.write(
            "The QA generator provides the possibility to generate question-context-answer triples from the provided documents that are used to evaluate hyperparameters and the corresponding RAG model in a consecutive step. You can either provide parameters for the generator through the drop-down menu below or by uploading a JSON file."
        )

        provide_qa_gen_form(valid_data)

        upload_files(
            context="qa_params",
            dropdown_msg="Upload JSON file",
            ext_list=["json"],
            file_path=get_label_dataset_gen_params_path(),
        )

        st.markdown("<br>" * 1, unsafe_allow_html=True)

        submit_button = st.button("Start evaluation dataset generation", key="SubmitQA")

        if submit_button:
            with st.spinner("Running generator..."):
                result = start_qa_gen()

                if "Success" in result:
                    st.success(result)
                else:
                    st.error(result)

    with tab2:
        st.write(
            "The Hyperparameter evaluator provides functionality of benchmarking RAG models with the corresponding parameters. During evaluation a LLM predicts an answer with the provided query and retrieved document chunks. With that we can calculate embedding similarities of label and predicted answers and ROUGE scores to provoide some metrics. We can also provide a LLM that is used for grading the predicted answers and the retrieved documents to extract even more metrics."
        )

        provide_hp_params_form(valid_data)

        upload_files(
            context="hp_params",
            dropdown_msg="Upload JSON file",
            ext_list=["json"],
            file_path=get_hyperparameters_path(),
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


def provide_hp_params_form(valid_data: dict):
    with st.expander("Hyperparameters settings"):
        attributes2 = {
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
                get_hyperparameters_path(),
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

            with st.spinner("Saving to file."):
                st.write("Saved to file. You've entered the following values:")
                # This is just a mockup to show you how to display the attributes.
                # You'll probably want to process or display them differently.
                st.write(attributes)
                # save json
                write_json(
                    attributes,
                    get_label_dataset_gen_params_path(),
                )
