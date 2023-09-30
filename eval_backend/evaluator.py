import glob
import dotenv
import tiktoken
import logging
import asyncio
import json

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

from utils import (
    load_document,
    split_data,
    get_retriever,
    get_qa_llm,
    generate_eval_set,
    write_json,
    aget_retrieved_documents,
)

from eval_metrics import (
    grade_embedding_similarity,
    grade_model_answer,
    grade_model_retrieval,
    grade_rouge,
)

logging.basicConfig(
    level=20,
    format="%(asctime)s - %(levelname)s - %(name)s:%(filename)s:%(lineno)d - %(message)s",
)

logger = logging.getLogger(__name__)

dotenv.load_dotenv(dotenv.find_dotenv(), override=True)

hyperparams_list = [
    {  # settings for qa generation, for the time being fixed across all evaluation runs
        "qa_generator_llm": ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        "length_function_for_qa_generation": lambda x: len(
            tiktoken.encoding_for_model("text-embedding-ada-002").encode(x)
        ),
        "generate_new_eval_set": False,
    },
    {
        # chunking stratety
        "chunk_size": 512,
        "chunk_overlap": 10,
        "length_function_default": len,
        # vectordb and retriever
        "embedding_model": OpenAIEmbeddings(model="text-embedding-ada-002"),
        "num_retrieved_docs": 3,
        # qa llm
        "retrieval_llm": ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        # answer grading llm
        "use_llm_grader": True,  # if you want to generate metrics with llm grading
        "grader_llm": ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        "grade_answer_prompt": "3cats_zero_shot",
        "grade_docs_prompt": "default",
    },
]


async def run_eval(
    chain="",
    retriever="",
    retriever_type="",
    k=3,
    grade_prompt="",
    file_path="./resources",
    eval_gt_path="./resources/eval_data.json",
):
    """After generating gt_dataset we need to calculate the metrics based on Chunking strategy, type of vectorstore, retriever (similarity search), QA LLM

        Dependencies of parameters and build order:
        generating chunks               | depends on chunking strategy
            |__ vector database         | depends on type von DB
                    |__  retriever      | depends on Embedding model
                            |__ QA LLM  | depends on LLM and digests retriever

            |__ grader: llm who grades generated answers

    TODO: State of the art retrieval with lots of additional LLM calls:
        - use "MMR" to filter out similiar document chunks during similarity search
        - Use SelfQueryRetriever to find information in query about specific context, e.g. querying specific document only
        - Use ContextualCompressionRetriever to compress or summarize the document chunks before serving them to QA-LLM

        - retriever object would like something like this, additional metadata info for SelfQueryRetriever has to be provided as well
                retriever = ContextualCompressionRetriever(
                    base_compressor=LLMChainExtractor.from_llm(llm),
                    base_retriever=SelfQueryRetriever.from_llm(..., earch_type = "mmr")
                )

        https://learn.deeplearning.ai/langchain-chat-with-your-data/lesson/5/retrieval
        https://learn.deeplearning.ai/langchain-chat-with-your-data/lesson/7/chat

    Args:
        chain (str, optional): _description_. Defaults to "".
        retriever (str, optional): _description_. Defaults to "".
        retriever_type (str, optional): _description_. Defaults to "".
        k (int, optional): _description_. Defaults to 3.
        grade_prompt (str, optional): _description_. Defaults to "".
        file_path (str, optional): _description_. Defaults to "./resources".
        eval_gt_path (str, optional): _description_. Defaults to "./resources/eval_data.json".
    """

    # generate qa pairs as evaluation dataset
    qa_gen_configs = hyperparams_list[0]

    try:
        with open(eval_gt_path, "r", encoding="utf-8") as file:
            logger.info("Evaluation dataset found, loading it.")
            gt_dataset = json.load(file)

    except FileNotFoundError:
        logger.warning("Evaluation dataset not found, creating it.")

        for file in glob.glob(f"{file_path}/*.pdf"):
            data = load_document(file)

            # TODO: len function for qa generation should be hyperparam?
            chunks = split_data(
                data=data,
                chunk_size=3000,
                chunk_overlap=0,
                length_function=qa_gen_configs["length_function_for_qa_generation"],
            )
            qa_pairs = generate_eval_set(
                llm=qa_gen_configs["qa_generator_llm"], chunks=chunks
            )

            # only get pairs with sufficiently long answer
            gt_dataset += [
                qa_pair for qa_pair in qa_pairs if len(qa_pair["answer"]) > 150
            ]

            write_json(gt_dataset, filename=eval_gt_path)

    for hyperparams in hyperparams_list[1:]:
        logger.info("Starting next hyperparams eval.")

        # create chunks of all provided documents
        # TODO: for now only the msg pdf in resources folder
        chunks_list = []
        for file in glob.glob(f"{file_path}/*.pdf"):
            data = load_document(file)
            chunks = split_data(
                data=data,
                chunk_size=hyperparams["chunk_size"],
                chunk_overlap=hyperparams["chunk_overlap"],
                length_function=hyperparams["length_function_default"],
            )

            chunks_list += chunks

        retriever = get_retriever(
            splits=chunks_list,
            embedding_model=hyperparams["embedding_model"],
            num_retrieved_docs=hyperparams["num_retrieved_docs"],
        )
        qa_llm = get_qa_llm(
            retriever=retriever, retrieval_llm=hyperparams["retrieval_llm"]
        )

        # dict[question, generated answer]
        predicted_answers = await asyncio.gather(
            *[qa_llm.acall(qa_pair) for qa_pair in gt_dataset]
        )

        sim = await grade_embedding_similarity(
            gt_dataset, predicted_answers, hyperparams["embedding_model"]
        )

        print(f"Similarity score: {sim}")

        rouge1, rouge2 = grade_rouge(gt_dataset, predicted_answers)

        print(f"Rouge1 and 2: {rouge1, rouge2}")

        cor, compr, read = grade_model_answer(
            gt_dataset,
            predicted_answers,
            hyperparams["grade_answer_prompt"],
            hyperparams["grader_llm"],
        )

        print(f"Answer grades (corr, compr, read): {cor, compr, read}")

        # dict[question, concatenated string of chunks retrieved]
        retrieved_docs = await asyncio.gather(
            *[aget_retrieved_documents(qa_pair, retriever) for qa_pair in gt_dataset]
        )

        accuracy = grade_model_retrieval(
            gt_dataset,
            retrieved_docs,
            hyperparams["grade_docs_prompt"],
            hyperparams["grader_llm"],
        )

        print(f"Retrieval accuracy: {accuracy}")


if __name__ == "__main__":
    asyncio.run(run_eval())
