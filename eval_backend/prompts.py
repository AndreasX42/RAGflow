from langchain.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)


template = """Please act as an impartial judge and evaluate the quality of the provided answer which attempts to answer the provided question based on a provided context.

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}

Your response should be as follows:

CORRECTNESS: (1,2,3,4 or 5) - grade 1 means the answer was completly incorrect, a higher grade towards 5 means the answer is more correct, does clarify more parts of the question and is more readable. The best grade is 5.
(line break)
JUSTIFICATION: (Without mentioning the student/teacher framing of this prompt, explain why the STUDENT ANSWER is Correct or Incorrect, identify potential sources of bias in the QUESTION, and identify potential sources of bias in the TRUE ANSWER. Use one or two sentences maximum. Keep the answer as concise as possible.)
"""

GRADE_ANSWER_PROMPT_FAST = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)


template = """Please act as an impartial judge and evaluate the quality of the provided answer which attempts to answer the provided question based on a provided context.

  You'll be given a function grading_function which you'll call for each provided context, question and answer to submit your reasoning and score for the correctness, comprehensiveness and readability of the answer. 

  Below is your grading rubric: 

- Correctness: Does the answer correctly answer the question.

- Comprehensiveness: How comprehensive is the answer, does it fully answer all aspects of the question and provide comprehensive explanation and other necessary information. 

- Readability: How readable is the answer, does it have redundant information or incomplete information that hurts the readability of the answer. Rate from 0 (completely unreadable) to 1 (highly readable)

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}

Your response should be as follows:

CORRECTNESS: (1,2,3,4 or 5) - grade 1 means completely incorrect and 5 means that the answer completley answers the question
COMPREHENSIVENESS: (1,2,3,4 or 5) - grade 1 completely incomprehensive and 5 highly comprehensive
READABILITY: (1,2,3,4 or 5) - grade 1 means completely unreadable and 5 highly readable
(line break)
JUSTIFICATION: (Without mentioning the student/teacher framing of this prompt, explain why the STUDENT ANSWER is Correct or Incorrect, identify potential sources of bias in the QUESTION, and identify potential sources of bias in the TRUE ANSWER. Use one or two sentences maximum. Keep the answer as concise as possible.)
"""

GRADE_ANSWER_PROMPT_3CATEGORIES_ZERO_SHOT_WITH_REASON = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)


template = """Please act as an impartial judge and evaluate the quality of the provided answer which attempts to answer the provided question based on a provided context.

  You'll be given a function grading_function which you'll call for each provided context, question and answer to submit your reasoning and score for the correctness, comprehensiveness and readability of the answer. 

  Below is your grading rubric: 

- Correctness: Does the answer correctly answer the question.

- Comprehensiveness: How comprehensive is the answer, does it fully answer all aspects of the question and provide comprehensive explanation and other necessary information. 

- Readability: How readable is the answer, does it have redundant information or incomplete information that hurts the readability of the answer. Rate from 0 (completely unreadable) to 1 (highly readable)

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}

Your response should be as follows, only 'METRIC: grade' for all three metrics. Do not provide any other information, you don't have to explain yourself.

CORRECTNESS: (1,2,3,4 or 5) - grade 1 means completely incorrect and 5 means that the answer completley answers the question
COMPREHENSIVENESS: (1,2,3,4 or 5) - grade 1 completely incomprehensive and 5 highly comprehensive
READABILITY: (1,2,3,4 or 5) - grade 1 means completely unreadable and 5 highly readable
"""

GRADE_ANSWER_PROMPT_3CATEGORIES_ZERO_SHOT = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)

template = """ 
    Given the question: \n
    {query}
    Here are some documents retrieved in response to the question: \n
    {result}
    And here is the answer to the question: \n 
    {answer}
    GRADING CRITERIA: We want to know if the question can be directly answered with the provided documents and without providing any additional outside sources. Does the retrieved documents make it possible to answer the question?

    Your response should be as follows, only 'GRADE: grade'. Do not provide any other information, you don't have to explain yourself.

    GRADE: (1,2,3,4 or 5) - grade 1 means it is impossible to answer the questions with the documents in any way, the more parts of the question you can answer the higher grade you should assign. If you can answer the question completley solely with the documents provided, the grade should be 5.
    """

GRADE_DOCS_PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)
