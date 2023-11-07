import chromadb
from chromadb.config import Settings
import os

API_HOST = os.environ.get("CHROMADB_HOST", "localhost")
API_PORT = os.environ.get("CHROMADB_PORT", 8000)


class ChromaClient:
    """Get a chromadb client as ContextManager."""

    def __init__(self):
        self.chroma_client = chromadb.HttpClient(
            host=API_HOST,
            port=API_PORT,
            settings=Settings(anonymized_telemetry=False),
        )

    def get_client(self):
        return self.chroma_client

    def __enter__(self):
        return self.chroma_client

    def __exit__(self, exc_type, exc_value, traceback):
        # TODO: self.chroma_client.stop()
        pass
