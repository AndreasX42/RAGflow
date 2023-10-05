from langchain.prompts.prompt import PromptTemplate

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""

QA_ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)
