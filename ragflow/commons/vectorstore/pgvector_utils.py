from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.sql import select
from langchain.vectorstores.pgvector import PGVector
from langchain.schema.embeddings import Embeddings

from ragflow.api import PGVECTOR_URL

TABLE = "langchain_pg_collection"
COL_NAME = "name"


def delete_collection(name: str) -> None:
    col = PGVector(
        collection_name=name,
        connection_string=PGVECTOR_URL,
        embedding_function=None,
    )

    col.delete_collection()


def create_collection(name: str, embedding: Embeddings) -> PGVector:
    col = PGVector(
        collection_name=name,
        connection_string=PGVECTOR_URL,
        embedding_function=embedding,
        pre_delete_collection=True,
    )

    col.create_collection()

    return col


def list_collections() -> list[str]:
    # Create an engine
    engine = create_engine(PGVECTOR_URL)

    # Reflect the specific table
    metadata = MetaData()
    table = Table(TABLE, metadata, autoload_with=engine)

    # Query the column
    query = select(table.c[COL_NAME])
    with engine.connect() as connection:
        results = connection.execute(query).fetchall()

    return results
