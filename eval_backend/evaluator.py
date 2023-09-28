import glob
import dotenv
import tiktoken

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

from .utils import (
    load_document,
    split_data,
    get_retriever,
    get_qa_llm,
    generate_eval_set,
    write_json,
)

dotenv.load_dotenv(dotenv.find_dotenv(), override=True)

hyperparams_list = [
    {
        # chunking stratety
        "chunk_size": 512,
        "chunk_overlap": 10,
        "length_function": lambda x: len(
            tiktoken.encoding_for_model("text-embedding-ada-002").encode(x)
        ),
        # vectordb and retriever
        "embedding_model": OpenAIEmbeddings(model="text-embedding-ada-002"),
        "num_retrieved_docs": 3,
        # qa llm
        "retrieval_llm": ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        # answer grading llm
        "grader_llm": ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        "use_grader": True,  # if you want to generate metrics with llm grading
    }
]


def run_eval(
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

    qa_gen_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    gt_dataset = []
    for file in glob.glob(f"{file_path}/*.pdf"):
        data = load_document(file)
        chunks = split_data(data=data, chunk_size=3000)
        qa_pairs = generate_eval_set(llm=qa_gen_llm, chunks=chunks)

        # only get pairs with sufficiently long answer
        gt_dataset += [qa_pair for qa_pair in qa_pairs if len(qa_pair["answer"]) > 150]

        write_json(gt_dataset, filename=eval_gt_path)

    for hyperparams in hyperparams_list:
        chunks_list = []
        for file in glob.glob(f"{file_path}/*.pdf"):
            data = load_document(file)
            chunks = split_data(
                data=data,
                chunk_size=hyperparams["chunk_size"],
                chunk_overlap=hyperparams["chunk_overlap"],
                length_function=hyperparams["length_function"],
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


run_eval()
