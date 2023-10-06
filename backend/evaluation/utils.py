from langchain.schema.retriever import BaseRetriever
from langchain.docstore.document import Document

import pandas as pd
import numpy as np
import re
import os


def extract_llm_metric(text: str, metric: str) -> float:
    """Utiliy function for extracting scores from LLM output from grading of generated answers and retrieved document chunks.

    Args:
        text (str): LLM result
        metric (str): name of metric

    Returns:
        number: the found score as integer or np.nan
    """
    match = re.search(f"{metric}: (\d+)", text)
    if match:
        return int(match.group(1))
    return np.nan


async def aget_retrieved_documents(
    qa_pair: dict[str, str], retriever: BaseRetriever
) -> dict:
    """Retrieve most similar documents to query asynchronously, postprocess the document chunks and return the qa pair with the retrieved documents string as result

    Args:
        qa_pair (dict[str, str]): _description_
        retriever (BaseRetriever): _description_

    Returns:
        list[dict]: _description_
    """
    query = qa_pair["question"]
    docs_retrieved = await retriever.aget_relevant_documents(query)

    retrieved_doc_text = "\n\n".join(
        f"Retrieved document {i}: {doc.page_content}"
        for i, doc in enumerate(docs_retrieved)
    )
    retrieved_dict = {
        "question": qa_pair["question"],
        "answer": qa_pair["answer"],
        "result": retrieved_doc_text,
    }

    return retrieved_dict


def process_retrieved_docs(qa_results: list[dict], hp_id: int) -> None:
    """For each list of Documents retrieved in QA_LLM call, we merge the documents page contents into a string. We store the id of the hyperparameters for later.

    Args:
        qa_results (list[dict]): _description_
        hp_id (int): _description_
    """

    # Each qa_result is a dict containing in "source_documents" the list of retrieved documents used to answer the query
    for qa_result in qa_results:
        retrieved_docs_str = "\n\n*************************\n\n".join(
            f"Retrieved document {i}: {doc.page_content}"
            for i, doc in enumerate(qa_result["source_documents"])
        )

        # key now stores the concatenated string of the page contents
        qa_result["retrieved_docs"] = retrieved_docs_str

        # id of hyperparameters needed later
        qa_result["hp_id"] = hp_id


def convert_element_to_df(element: dict):
    """Function to convert each tuple element of result to dataframe."""
    df = pd.DataFrame(element)
    df = pd.concat([df, pd.json_normalize(df["metadata"])], axis=1)
    df = df.drop(columns=["metadata", "question", "answer"])
    df = df.rename(columns={"result": "predicted_answer", "id": "qa_id"})
    df["hp_id"] = df["hp_id"].astype(int)

    return df


def write_generated_data_to_csv(
    predicted_answers: list[dict],
    hp_runs_data_path: str,
) -> None:
    """Write predicted answerd and retrieved documents to csv."""
    # Create the dataframe
    df = pd.concat(
        [convert_element_to_df(element) for element in predicted_answers], axis=0
    )

    if os.path.exists(hp_runs_data_path):
        base_df = pd.read_csv(hp_runs_data_path)
    else:
        base_df = pd.DataFrame()

    # Concatenate the DataFrames along rows while preserving shared columns
    result = pd.concat(
        [
            base_df,
            df[["hp_id", "predicted_answer", "retrieved_docs", "qa_id", "source"]],
        ],
        axis=0,
    )

    # Write df to csv
    result.sort_values(by=["hp_id"]).to_csv(hp_runs_data_path, index=False)
