import chromadb
from chromadb.config import Settings


class ChromaClient:
    """Get a chromadb client as ContextManager."""

    def __init__(self):
        self.chroma_client = chromadb.HttpClient(
            host="localhost",
            port=8000,
            settings=Settings(anonymized_telemetry=False),
        )

    def __enter__(self):
        return self.chroma_client

    def __exit__(self, exc_type, exc_value, traceback):
        self.chroma_client.stop()
