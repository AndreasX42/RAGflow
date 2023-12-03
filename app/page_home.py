import streamlit as st


# Define pages and their functions
def page_home():
    st.title("Welcome to RAGflow!")

    tab1, tab2, tab3 = st.tabs(["The App", "Getting Started", "About"])

    with tab1:
        # Introductory text
        st.write(
            """
        RAGflow is an advanced application framework tailored to streamline the construction and evaluation processes for Retrieval Augmented Generation (RAG) systems in question-answer contexts. Here's a brief rundown of its functionality:
        """
        )

        # Functionality points
        st.header("Key Features & Functionalities")

        # 1. Document Store Integration
        st.subheader("1. Document Store Integration")
        st.write(
            """
        RAGflow starts by interfacing with a variety of document stores, enabling users to select from a diverse range of data sources.
        """
        )

        # 2. Dynamic Parameter Selection
        st.subheader("2. Dynamic Parameter Selection")
        st.write(
            """
        Through an intuitive interface, users can customize various parameters, including document splitting strategies, embedding model choices, and question-answering LLMs. These parameters influence how the app splits, encodes, and processes data.
        """
        )

        # 3. Automated Dataset Generation
        st.subheader("3. Automated Dataset Generation")
        st.write(
            """
        One of RAGflow's core strengths is its capability to auto-generate datasets. This feature allows for a seamless transition from raw data to a structured format ready for processing.
        """
        )

        # 4. Advanced Retrieval Mechanisms
        st.subheader("4. Advanced Retrieval Mechanisms")
        st.write(
            """
        Incorporating state-of-the-art retrieval methods, RAGflow employs techniques like "MMR" for filtering and the SelfQueryRetriever for targeted data extraction. This ensures that the most relevant document chunks are presented for question-answering tasks.
        """
        )

        # 5. Integration with External Platforms
        st.subheader("5. Integration with External Platforms")
        st.write(
            """
        RAGflow is designed to work in tandem with platforms like Anyscale, MosaicML, and Replicate, offering users extended functionalities and integration possibilities.
        """
        )

        # 6. Evaluation & Optimization
        st.subheader("6. Evaluation & Optimization")
        st.write(
            """
        At its core, RAGflow is built to optimize performance. By performing grid searches across the selected parameter space, it ensures that users achieve the best possible results for their specific configurations.
        """
        )

        # 7. Interactive Feedback Loop
        st.subheader("7. Interactive Feedback Loop")
        st.write(
            """
        As users interact with the generated question-answer systems, RAGflow offers feedback mechanisms, allowing for continuous improvement and refinement based on real-world results.
        """
        )

        # Conclusion
        st.write(
            """
        Dive into RAGflow, set your parameters, and watch as it automates and optimizes the intricate process of building robust, data-driven question-answering systems!
        """
        )

    with tab2:
        # Functional stages
        st.header("Using This App: A Step-by-Step Guide")

        st.write(
            """
        1. Account Setup:
        Log in or register.

        2. API Key Submission:
        Supply all the required API keys, such as those for OpenAI. The keys needed will vary based on the LLMs and services you intend to use.

        3. Document Upload:
        Upload the documents for which you aim to create a RAG application.

        4. Parameter Configuration:
        Input the necessary parameters directly or by uploading a JSON file. This is essential for both the evaluation dataset generation and the hyperparameter assessment.

        5. Dashboard Analysis:
        Navigate to the Dashboard to inspect hyperparameter metrics and view the generated data.

        6. File Management:
        The File Manager allows you to delete or download files within your directory for added flexibility.       
        """
        )

    with tab3:
        st.subheader("About Me")
        st.write(
            """
        Hello! I'm a passionate developer and AI enthusiast working on cutting-edge projects to leverage the power of machine learning. 
        My latest project, RAGflow, focuses on building and evaluating Retrieval Augmented Generation (RAG) systems. 
        Curious about my work? Check out my [GitHub repository](https://github.com/AndreasX42/RAGflow) for more details!
        """
        )
