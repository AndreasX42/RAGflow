from langchain.prompts import PromptTemplate

template = """
    Given the question:
    {query}
    
    Here are some documents retrieved in response to the question:
    {result}
    
    And here is the answer to the question:
    {answer}
    
    GRADING CRITERIA: We want to know if the question can be directly answered with the provided documents and without providing any additional outside sources. Does the retrieved documents make it possible to answer the question?

    Your response should be as follows without providing any additional information:

    GRADE: (1,2,3,4 or 5) - grade 1 means it is impossible to answer the questions with the documents in any way, the more parts of the question you can answer the higher the grade you should assign. If you can answer the question completley solely with the documents provided, the grade should be 5.
    """

GRADE_RETRIEVER_PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)
