# RAGflow: Build optimized and robust LLM applications

[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/6FfqBzs4fBDyTPvBNqnq5x/8HU8omXUEUaEgrpWMj271K/tree/main.svg?style=shield&circle-token=545d0058e25f4566f54a9282ef976f6a8a77b327)](https://app.circleci.com/pipelines/circleci/6FfqBzs4fBDyTPvBNqnq5x)

RAGflow provides tools for constructing and evaluating Retrieval Augmented Generation (RAG) systems, empowering developers to craft efficient question-answer applications leveraging LLMs.

# 📋 TODO
- Write unit and integration tests for CI
- Implement better Login/Authentication in Streamlit app
- Set up CD onto Cloud environment (Kubernetes)
- Optimize some design choices (shared volumes should be some cloud storage)
    
# 🌟 Key Features & Functionalities
- `Document Store Integration` Provide documents in formats like pdf and docx as knowledge base.
- `Dynamic Parameter Selection` Customize parameters such as document splitting strategies, embedding model, and question-answering LLMs for evaluations.
- `Automated Dataset Generation` Automatically generates question answer pairs from the provided documents as evaluation dataset to evaluate each parameterized RAG system.
- `Evaluation & Optimization` Optimize performance using grid searches across parameter spaces.
- `Advanced Retrieval Mechanisms` Employ techniques like "MMR" and the SelfQueryRetriever for optimal data extraction.
- `Integration with External Platforms` Collaborate with platforms like Anyscale, MosaicML, and Replicate for enhanced functionalities and state-of-the-art LLM models.
- `Interactive Feedback Loop` Refine and improve your RAG system with interactive feedback based on real-world results.
    
# 🚀 Workflow
- Automatic Generation of Question-Answer Pairs\
Begin with RAGflow's capability to generate relevant question-answer pairs from provided documents.
Hyperparameter Evaluation
- Evaluate provided hyperparameters \
After generating Q&A pairs, dive deep into hyperparameter evaluation. Provide your hyperparameters, let RAGflow evaluate their efficacy, and obtain insights for crafting robust RAG systems.

# 📖 Dive Deeper
For a more interactive and comprehensive guide on RAGflow's functionalities and usage, check out our Streamlit frontend application. Here, we provide various detailed insights, set your parameters, and watch RAGflow automate the process of constructing RAG systems.

# 🛠️ Development
Directory Structure
- `/.circleci` CircleCI integration config for CI/CD pipeline.
- `/app` Frontend components and resources in Streamlit.
- `/backend` Backend services and APIs.
- `/tests` Test scripts and test data.
- `/tmp` Temporary storage.
- `/vectorstore` ChromaDB component.
- `docker-compose.yml` Docker configurations for setting up the local environment.
    
# 🚀 Getting Started
Setup
- Checkout repository
- Start application with ‘docker-compose up --build’
    
# 🌐 Links & Resources
- TBA
    
# 🙋 About the Developer
Hi, I'm Andreas! As a devoted developer and AI aficionado, I've always been drawn to leveraging the prowess of machine learning. RAGflow is my endeavor to simplify and optimize the creation of Retrieval Augmented Generation systems. Keen to explore? Dive into my GitHub repository for an in-depth look!
