import json
import asyncio
from uuid import UUID
import pandas as pd

from langchain.vectorstores.chroma import Chroma
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import LLMResult, Document

from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
)

from ragflow.commons.chroma import ChromaClient
from ragflow.commons.configurations import Hyperparameters

from typing import Any, List

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
    llm = getOrCreateChatModel(hp_id, hyperparameters_results_path, user_id, api_keys)

    # attach callback to llm
    llm.combine_docs_chain.llm_chain.llm.streaming = True
    llm.combine_docs_chain.llm_chain.llm.callbacks = [stream_it]

    logger.error(llm)

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
    llm = getOrCreateChatModel(hp_id, hyperparameters_results_path, user_id, api_keys)

    return llm(query)


async def get_docs(
    hp_id: int,
    hyperparameters_results_path: str,
    user_id: str,
    api_keys: dict,
    query: str,
) -> List[Document]:
    # get or load llm model
    llm = getOrCreateChatModel(hp_id, hyperparameters_results_path, user_id, api_keys)

    return await llm.retriever.retriever.aget_relevant_documents(query)


def getOrCreateChatModel(
    hp_id: int, hyperparameters_results_path: str, user_id: str, api_keys: dict
) -> None:
    # if model has not been loaded yet
    if (
        user_id not in chats_cache
        or hp_id not in chats_cache[user_id]
        or not isinstance(chats_cache[user_id][hp_id], ConversationalRetrievalChain)
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

        index = Chroma(
            client=ChromaClient().get_client(),
            collection_name=collection.name,
            collection_metadata=collection.metadata,
            embedding_function=hp.embedding_model,
        )

        # baseline retriever built from vectorstore collection
        retriever = index.as_retriever(
            search_type=hp.search_type.value, search_kwargs={"k": hp.num_retrieved_docs}
        )

        # LLM chain for generating new question from user query and chat history
        question_generator = LLMChain(llm=hp.qa_llm, prompt=CONDENSE_QUESTION_PROMPT)

        # llm that answers the newly generated condensed question
        doc_qa_chain = load_qa_chain(
            hp.qa_llm.copy(), chain_type="stuff", prompt=QA_PROMPT
        )

        # advanced retriever using multiple similar queries
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever, llm=hp.qa_llm
        )

        # memory object to store the chat history
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history", k=5, return_messages=True, output_key="result"
        )

        qa_llm = ConversationalRetrievalChain.from_llm(
            retriever=multi_query_retriever,
            combine_docs_chain=doc_qa_chain,
            question_generator=question_generator,
            return_generated_question=False,
            return_source_documents=False,
            memory=memory,
            output_key="result",
        )

        # cache llm
        if user_id not in chats_cache:
            chats_cache[user_id] = {}

        chats_cache[user_id][hp_id] = qa_llm

    # model is already loaded
    return chats_cache[user_id][hp_id]
