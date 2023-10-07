import streamlit as st
import json


def page_params():
    st.title("Parameters Page")
    st.subheader("Provide parameters for building and evaluating the system.")

    # Dropdown for entering evaluation set
    with st.expander("QA Generator settings"):
        attributes = {
            "chunk_size": st.number_input("Chunk Size", value=512),
            "chunk_overlap": st.number_input("Chunk Overlap", value=10),
            "length_function_name": st.text_input("Length Function Name", "len"),
            "qa_generator_llm": st.text_input("QA LLM", "gpt-3.5-turbo"),
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
            st.write("You've entered the following values:")
            # This is just a mockup to show you how to display the attributes.
            # You'll probably want to process or display them differently.
            st.write(attributes)

    # Dropdown for entering hyperparameters
    with st.expander("Hyperparameter settings"):
        attributes2 = {
            "id": st.number_input("ID", value=0, key="id for this HP config"),
            "chunk_size": st.number_input("Chunk Size", value=512, key="chunk size"),
            "chunk_overlap": st.number_input(
                "Chunk Overlap", value=10, key="chunk overlap"
            ),
            "length_function_name": st.text_input(
                "Length Function Name (Used for chunking)",
                "len",
                key="length function for document chunking.",
            ),
            "num_retrieved_docs": st.number_input(
                "Number of Docs the retriever should return",
                value=3,
                key="number of docs to retrieve",
            ),
            "search_type": st.selectbox("Search Type", ["mmr", "other_type"]),
            "embedding_model": st.text_input(
                "Embedding Model",
                "text-embedding-ada-002",
                key="Name of embedding model.",
            ),
            "qa_llm": st.text_input(
                "QA LLM", "gpt-3.5-turbo", key="Name of LLM for QA task."
            ),
        }

        use_llm_grader = st.checkbox(
            "Use LLM Grader", value=False, key="use_llm_grader_checkbox"
        )

        if use_llm_grader:
            attributes2["grade_answer_prompt"] = st.selectbox(
                "Grade Answer Prompt",
                ["few_shot", "zero_shot", "fast"],
                key="Type of prompt to use for answer grading.",
            )
            attributes2["grade_docs_prompt"] = st.text_input(
                "Grade Documents Prompt",
                "default",
                key="Type of prompt to grade retrieved document chunks.",
            )
            attributes2["grader_llm"] = st.text_input(
                "Grading LLM", "gpt-3.5-turbo", key="Name of LLM for grading."
            )

        submit_button = st.button("Submit", key="Submit2")
        if submit_button:
            attributes2["use_llm_grader"] = use_llm_grader
            st.write("You've entered the following values:")
            # This is just a mockup to show you how to display the attributes.
            # You'll probably want to process or display them differently.
            st.write(attributes2)

    st.text("")
    uploaded_file = st.file_uploader(
        label="Upload a JSON file to provide a list of hyperparameters instead.",
        type=["json"],
    )

    if uploaded_file is not None:
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

    st.code(
        """
    {
        "id": 0,
        "chunk_size": 1024,
        "chunk_overlap": 10,
        "num_retrieved_docs": 3,
        "length_function_name": "len",
        "search_type": "mmr",
        "embedding_model": "text-embedding-ada-002",
        "qa_llm": "gpt-3.5-turbo",
        "use_llm_grader": false,
        "grade_answer_prompt": "few_shot",
        "grade_docs_prompt": "default",
        "grader_llm": "gpt-3.5-turbo",
    }
    
    """
    )
