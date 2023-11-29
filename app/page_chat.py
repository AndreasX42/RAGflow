import pandas as pd
import streamlit as st
from utils import *
import os


def page_chat():
    # Title of the page
    st.subheader("Chat with a RAG model from a hyperparameter evaluation")

    if display_user_login_warning():
        return

    if not os.path.exists(get_hyperparameters_results_path()):
        st.warning("No hyperparameter results available. Run some evaluation.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I help you?"}
        ]

    with st.expander("View hyperparameter results"):
        df = load_hp_results()
        showData = st.multiselect(
            "Filter: ",
            df.columns,
            default=[
                "id",
                "chunk_size",
                "chunk_overlap",
                "length_function_name",
                "num_retrieved_docs",
                "search_type",
                "similarity_method",
                "use_llm_grader",
            ],
        )
        st.dataframe(df[showData], use_container_width=True)

    # select chat model from hyperparam run id
    hp_id = 0
    hp_id = st.selectbox(
        "Select chat model from hyperparameter evaluations", options=list(df.id)
    )
    # Check if hp_id changed and clear chat history if it did
    if (
        "hp_id" in st.session_state
        and hp_id != st.session_state.hp_id
        or "hp_id" not in st.session_state
    ):
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I help you?"}
        ]
        # Update the session state with the new hp_id
    st.session_state.hp_id = hp_id

    st.markdown("<br>" * 1, unsafe_allow_html=True)
    st.write("Chat with the chosen parametrized RAG")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided query
    if query := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            # get and display answer
            answer, source_docs = get_rag_response_stream(hp_id, query)

            # display retrieved documents
            if "I don't know".lower() not in answer.lower() and source_docs is not None:
                display_documents(source_docs)

        message = {"role": "assistant", "content": answer}
        st.session_state.messages.append(message)


def display_documents(documents: list[dict]):
    for idx, doc in enumerate(documents["source_documents"]):
        with st.expander(f"Document {idx + 1}"):
            st.text(f"Source: {os.path.basename(doc['metadata']['source'])}")
            st.text(
                f"Index location: {doc['metadata']['start_index']}-{doc['metadata']['end_index']}"
            )
            st.text_area("Content", value=doc["page_content"], height=150)


def retrieve_source_documents(hp_id: int, query: str):
    documents = get_docs_from_query(hp_id, query)
    display_documents(documents)


def load_hp_results() -> pd.DataFrame:
    with open(get_hyperparameters_results_path(), encoding="utf-8") as file:
        hp_data = json.load(file)

    df = pd.DataFrame(hp_data)
    df["id"] = df["id"].astype(int)

    # Flatten the "scores" sub-dictionary
    scores_df = pd.json_normalize(df["scores"])

    # Combine the flattened scores DataFrame with the original DataFrame
    df = pd.concat(
        [df[["id"]], df.drop(columns=["id", "scores"], axis=1), scores_df], axis=1
    )

    # Print the resulting DataFrame
    return df


def display_rag_response_stream(hp_id: int, query: str):
    # Start a separate thread for fetching stream data
    if "rag_response_stream_data" not in st.session_state:
        st.session_state["rag_response_stream_data"] = "Starting stream...\n"

    get_rag_response_stream(hp_id, query)
