import chromadb
from chromadb.config import Settings
import os

API_HOST = os.environ.get("CHROMADB_HOST")
API_PORT = os.environ.get("CHROMADB_PORT")


class ChromaClient:
    """Get a chromadb client as ContextManager."""

    def __init__(self):
        self.chroma_client = chromadb.HttpClient(
            host=API_HOST,
            port=API_PORT,
            settings=Settings(anonymized_telemetry=False),
        )

    def __enter__(self):
        return self.chroma_client

    def __exit__(self, exc_type, exc_value, traceback):
        self.chroma_client.stop()
