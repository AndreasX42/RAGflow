# RAGflow: Build optimized and robust LLM applications

[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/6FfqBzs4fBDyTPvBNqnq5x/8HU8omXUEUaEgrpWMj271K/tree/main.svg?style=shield&circle-token=545d0058e25f4566f54a9282ef976f6a8a77b327)](https://app.circleci.com/pipelines/circleci/6FfqBzs4fBDyTPvBNqnq5x)

RAGflow provides tools for constructing and evaluating Retrieval Augmented Generation (RAG) systems, empowering developers to craft efficient question-answer applications leveraging LLMs. The stack consists of

`Language` [Python](https://www.python.org/)\
`Frameworks for LLMs` [LangChain](https://www.langchain.com/) [OpenAI](https://www.openai.com/) [Hugging Face](https://huggingface.co/)\
`Framework for API` [FastAPI](https://fastapi.tiangolo.com/)\
`Databases` [ChromaDB](https://www.trychroma.com/) [Postgres](https://www.postgresql.org/)\
`Frontend` [Streamlit](https://www.streamlit.io/)\
`CI/CD` [Docker](https://www.docker.com/) [Kubernetes](https://kubernetes.io/) [CircleCI](https://circleci.com/) [GKE](https://cloud.google.com/kubernetes-engine)

# 🚀 Getting Started

- CircleCI pushes the Docker images after each successful build to
  - https://hub.docker.com/u/andreasx42
- Google Kubernetes Engine cluster could be available on
  - http://35.239.36.15/
- Checkout repository
  - Start application with ‘docker-compose up --build’
    - Application should be available on localhost:8501.
    - Backend API documentation is available on localhost:8080/docs
  - Use Kubernetes with 'kubectl apply -f k8s' to deploy locally
    - Application should be available directly on localhost/
    - For backend API access we use nginx routing with localhost/api/\*
    - Be aware to check deployment configs for image versions

# 📖 What is Retrievel Augmented Generation (RAG)?

<img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*Jq9bEbitg1Pv4oASwEQwJg.png" alt="Description" width="850"/>
<p align="left"><a href="https://towardsdatascience.com/rag-vs-finetuning-which-is-the-best-tool-to-boost-your-llm-application-94654b1eaba7">Source</a></p>

In RAG, when a user query is received, relevant documents or passages are retrieved from a massive corpus, i.e. a document store. These retrieved documents are then provided as context to a generative model, which synthesizes a coherent response or answer using both the input query and the retrieved information. This approach leverages the strengths of both retrieval-based and generative systems, aiming to produce accurate and well-formed responses by drawing from vast amounts of textual data.

# 🚀 Workflow of RAGflow

- Automatic Generation of Question-Answer Pairs\
  Begin with RAGflow's capability to generate relevant question-answer pairs from provided documents which is used as an evaluation dataset to evaluate RAG systems.
  Hyperparameter Evaluation
- Evaluate provided hyperparameters \
  After generating Q&A pairs, dive into hyperparameter evaluation. Provide your hyperparameters, let RAGflow evaluate their efficacy, and obtain insights for crafting robust RAG systems.
  This approach allows you to select efficient document splitting strategies, language and embedding models which could be further finetuned with respect to your document store.

Here is a schematic overview:

![schematics](https://github.com/AndreasX42/RAGflow/assets/141482745/8ea78a21-8224-4baf-a441-dc4aa8249762)

# 🌟 Key Features & Functionalities

- `Document Store Integration` Provide documents in formats like pdf and docx as knowledge base.
- `Dynamic Parameter Selection` Customize parameters such as document splitting strategies, embedding model, and question-answering LLMs for evaluations.
- `Automated Dataset Generation` Automatically generates question answer pairs from the provided documents as evaluation dataset to evaluate each parameterized RAG system.
- `Evaluation & Optimization` Optimize performance using grid searches across parameter spaces.
- `Advanced Retrieval Mechanisms` Employ techniques like "MMR" and the SelfQueryRetriever for optimal data extraction.
- `Integration with External Platforms` Collaborate with platforms like Anyscale, MosaicML, and Replicate for enhanced functionalities and state-of-the-art LLM models.
- `Interactive Feedback Loop` Refine and improve your RAG system with interactive feedback based on real-world results.

# 🛠️ Development

Directory Structure

- `/.circleci` CircleCI integration config for CI/CD pipeline.
- `/app` Frontend components and resources in Streamlit.
- `/ragflow` Backend services and APIs.
- `/tests` Test scripts and test data.
- `/notebooks` Jupyter notebooks with different experiments.
- `/resources` Data storage.
- `/vectorstore` ChromaDB component.

# 🌐 Links & Resources

- TBA
