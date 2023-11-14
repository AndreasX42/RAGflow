import json
import asyncio
from uuid import UUID
import pandas as pd

from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.schema import LLMResult, Document

from ragflow.commons.chroma import ChromaClient
from ragflow.commons.configurations import Hyperparameters
from ragflow.utils import get_qa_llm

from typing import Any, List, Optional

import logging

logger = logging.getLogger(__name__)

chats_cache = {}


class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    # content: str = ""
    # final_answer: bool = True

    def __init__(self) -> None:
        super().__init__()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.queue.put_nowait(token)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.done.set()


async def aquery_chat(
    hp_id: int,
    hyperparameters_results_path: str,
    user_id: str,
    api_keys: dict,
    query: str,
    stream_it: AsyncCallbackHandler,
):
    llm = getOrCreateModel(hp_id, hyperparameters_results_path, user_id, api_keys)

    # attach callback to llm
    llm.combine_documents_chain.llm_chain.llm.streaming = True
    llm.combine_documents_chain.llm_chain.llm.callbacks = [stream_it]

    await llm.acall(query)


async def create_gen(
    hp_id: int,
    hyperparameters_results_path: str,
    user_id: str,
    api_keys: dict,
    query: str,
    stream_it: AsyncCallbackHandler,
):
    task = asyncio.create_task(
        aquery_chat(
            hp_id, hyperparameters_results_path, user_id, api_keys, query, stream_it
        )
    )
    async for token in stream_it.aiter():
        yield token
    await task


def query_chat(
    hp_id: int,
    hyperparameters_results_path: str,
    user_id: str,
    api_keys: dict,
    query: str,
) -> dict:
    # get or load llm model
    llm = getOrCreateModel(hp_id, hyperparameters_results_path, user_id, api_keys)

    return llm(query)


async def get_docs(
    hp_id: int,
    hyperparameters_results_path: str,
    user_id: str,
    api_keys: dict,
    query: str,
) -> List[Document]:
    # get or load llm model
    llm = getOrCreateModel(hp_id, hyperparameters_results_path, user_id, api_keys)

    return await llm.retriever.aget_relevant_documents(query)


def getOrCreateModel(
    hp_id: int, hyperparameters_results_path: str, user_id: str, api_keys: dict
) -> None:
    # if model has not been loaded yet
    if (
        user_id not in chats_cache
        or hp_id not in chats_cache[user_id]
        or not isinstance(chats_cache[user_id][hp_id], RetrievalQA)
    ):
        # load hyperparameter results
        with open(hyperparameters_results_path, encoding="utf-8") as file:
            hp_data = json.load(file)

        df = pd.DataFrame(hp_data)

        # check that hp_id really exists in results
        if hp_id not in df.id.values:
            raise NotImplementedError("Could not find requested hyperparameter run id.")

        with ChromaClient() as client:
            for col in client.list_collections():
                if col.name == f"userid_{user_id[:8]}_hpid_{hp_id}":
                    collection = col
                    break

        # check that vectorstore contains collection for hp id
        if not collection:
            raise NotImplementedError(
                "Could not find data in vectorstore for requested hyperparameter run id."
            )

        # create retriever and llm from collection
        hp_data = df[df.id == hp_id].iloc[0].to_dict()
        for key in ["id", "scores", "timestamp"]:
            hp_data.pop(key)

        hp = Hyperparameters.from_dict(
            input_dict=hp_data,
            hp_id=hp_id,
            api_keys=api_keys,
        )

        vs = Chroma(
            client=ChromaClient().get_client(),
            collection_name=collection.name,
            collection_metadata=collection.metadata,
            embedding_function=hp.embedding_model,
        )

        retriever = vs.as_retriever(
            search_type=hp.search_type.value, search_kwargs={"k": hp.num_retrieved_docs}
        )

        # build RetrievalQA from retriever and llm
        qa_llm = get_qa_llm(retriever, hp.qa_llm, return_source_documents=True)

        # cache llm
        if user_id not in chats_cache:
            chats_cache[user_id] = {}

        chats_cache[user_id][hp_id] = qa_llm

    # model is already loaded
    return chats_cache[user_id][hp_id]
