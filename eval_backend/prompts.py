from langchain.prompts import PromptTemplate

template = """You are a teacher grading a quiz. 
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either Correct or Incorrect.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: Correct or Incorrect here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is Incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:

Your response should be as follows:

GRADE: (Correct or Incorrect)
(line break)
JUSTIFICATION: (Without mentioning the student/teacher framing of this prompt, explain why the STUDENT ANSWER is Correct or Incorrect. Use one or two sentences maximum. Keep the answer as concise as possible.)
"""

GRADE_ANSWER_PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)

template = """You are a teacher grading a quiz. 
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either Correct or Incorrect.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: Correct or Incorrect here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is Incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:"""

GRADE_ANSWER_PROMPT_FAST = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)

template = """You are a teacher grading a quiz. 
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either Correct or Incorrect.
You are also asked to identify potential sources of bias in the question and in the true answer.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: Correct or Incorrect here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is Incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:

Your response should be as follows:

GRADE: (Correct or Incorrect)
(line break)
JUSTIFICATION: (Without mentioning the student/teacher framing of this prompt, explain why the STUDENT ANSWER is Correct or Incorrect, identify potential sources of bias in the QUESTION, and identify potential sources of bias in the TRUE ANSWER. Use one or two sentences maximum. Keep the answer as concise as possible.)
"""

GRADE_ANSWER_PROMPT_BIAS_CHECK = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)

template = """You are assessing a submitted student answer to a question relative to the true answer based on the provided criteria: 
    
    ***
    QUESTION: {query}
    ***
    STUDENT ANSWER: {result}
    ***
    TRUE ANSWER: {answer}
    ***
    Criteria: 
      relevance:  Is the submission referring to a real quote from the text?"
      conciseness:  Is the answer concise and to the point?"
      correct: Is the answer correct?"
    ***
    Does the submission meet the criterion? First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print "Correct" or "Incorrect" (without quotes or punctuation) on its own line corresponding to the correct answer.
    Reasoning:
"""

GRADE_ANSWER_PROMPT_OPENAI = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)

template = """ 
    Given the question: \n
    {query}
    Here are some documents retrieved in response to the question: \n
    {result}
    And here is the answer to the question: \n 
    {answer}
    Criteria: 
      relevance: Are the retrieved documents relevant to the question and do they support the answer?"
    Do the retrieved documents meet the criterion? Print "Correct" (without quotes or punctuation) if the retrieved context are relevant or "Incorrect" if not (without quotes or punctuation) on its own line. """

GRADE_DOCS_PROMPT_FAST = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)

template = """ 
    Given the question: \n
    {query}
    Here are some documents retrieved in response to the question: \n
    {result}
    And here is the answer to the question: \n 
    {answer}
    Criteria: 
      relevance: Are the retrieved documents relevant to the question and do they support the answer?"

    Your response should be as follows:

    GRADE: (Correct or Incorrect, depending if the retrieved documents meet the criterion)
    (line break)
    JUSTIFICATION: (Write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Use one or two sentences maximum. Keep the answer as concise as possible.)
    """

GRADE_DOCS_PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)


template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

template = """
### Human
You are question-answering assistant tasked with answering questions based on the provided context. 

Here is the question: \
{question}

Use the following pieces of context to answer the question at the end. Use three sentences maximum. \
{context}

### Assistant
Answer: Think step by step. """
QA_CHAIN_PROMPT_LLAMA = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)


template = """Please act as an impartial judge and evaluate the quality of the provided answer which attempts to answer the provided question based on a provided context.

  You'll be given a function grading_function which you'll call for each provided context, question and answer to submit your reasoning and score for the correctness, comprehensiveness and readability of the answer. 

  Below is your grading rubric: 

- Correctness: If the answer correctly answer the question, below are the details for different scores:

  - Score 0: the answer is completely incorrect, doesn’t mention anything about the question or is completely contrary to the correct answer.

      - For example, when asked “How to terminate a databricks cluster”, the answer is empty string, or content that’s completely irrelevant, or sorry I don’t know the answer.

  - Score 1: the answer provides some relevance to the question and answers one aspect of the question correctly.

      - Example:

          - Question: How to terminate a databricks cluster

          - Answer: Databricks cluster is a cloud-based computing environment that allows users to process big data and run distributed data processing tasks efficiently.

          - Or answer:  In the Databricks workspace, navigate to the "Clusters" tab. And then this is a hard question that I need to think more about it

  - Score 2: the answer mostly answer the question but is missing or hallucinating on one critical aspect.

      - Example:

          - Question: How to terminate a databricks cluster”

          - Answer: “In the Databricks workspace, navigate to the "Clusters" tab.

          Find the cluster you want to terminate from the list of active clusters.

          And then you’ll find a button to terminate all clusters at once”

  - Score 3: the answer correctly answer the question and not missing any major aspect

      - Example:

          - Question: How to terminate a databricks cluster

          - Answer: In the Databricks workspace, navigate to the "Clusters" tab.

          Find the cluster you want to terminate from the list of active clusters.

          Click on the down-arrow next to the cluster name to open the cluster details.

          Click on the "Terminate" button. A confirmation dialog will appear. Click "Terminate" again to confirm the action.”

- Comprehensiveness: How comprehensive is the answer, does it fully answer all aspects of the question and provide comprehensive explanation and other necessary information. Below are the details for different scores:

  - Score 0: typically if the answer is completely incorrect, then the comprehensiveness is also zero score.

  - Score 1: if the answer is correct but too short to fully answer the question, then we can give score 1 for comprehensiveness.

      - Example:

          - Question: How to use databricks API to create a cluster?

          - Answer: First, you will need a Databricks access token with the appropriate permissions. You can generate this token through the Databricks UI under the 'User Settings' option. And then (the rest is missing)

  - Score 2: the answer is correct and roughly answer the main aspects of the question, but it’s missing description about details. Or is completely missing details about one minor aspect.

      - Example:

          - Question: How to use databricks API to create a cluster?

          - Answer: You will need a Databricks access token with the appropriate permissions. Then you’ll need to set up the request URL, then you can make the HTTP Request. Then you can handle the request response.

      - Example:

          - Question: How to use databricks API to create a cluster?

          - Answer: You will need a Databricks access token with the appropriate permissions. Then you’ll need to set up the request URL, then you can make the HTTP Request. Then you can handle the request response.

  - Score 3: the answer is correct, and covers all the main aspects of the question

- Readability: How readable is the answer, does it have redundant information or incomplete information that hurts the readability of the answer.

  - Score 0: the answer is completely unreadable, e.g. fully of symbols that’s hard to read; e.g. keeps repeating the words that it’s very hard to understand the meaning of the paragraph. No meaningful information can be extracted from the answer.

  - Score 1: the answer is slightly readable, there are irrelevant symbols or repeated words, but it can roughly form a meaningful sentence that cover some aspects of the answer.

      - Example:

          - Question: How to use databricks API to create a cluster?

          - Answer: You you  you  you  you  you  will need a Databricks access token with the appropriate permissions. And then then you’ll need to set up the request URL, then you can make the HTTP Request. Then Then Then Then Then Then Then Then Then

  - Score 2: the answer is correct and mostly readable, but there is one obvious piece that’s affecting the readability (mentioning of irrelevant pieces, repeated words)

      - Example:

          - Question: How to terminate a databricks cluster

          - Answer: In the Databricks workspace, navigate to the "Clusters" tab.

          Find the cluster you want to terminate from the list of active clusters.

          Click on the down-arrow next to the cluster name to open the cluster details.

          Click on the "Terminate" button…………………………………..

          A confirmation dialog will appear. Click "Terminate" again to confirm the action.

  - Score 3: the answer is correct and reader friendly, no obvious piece that affect readability.

- Then final rating:

    - Ratio: 60% correctness + 20% comprehensiveness + 20% readability

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is Incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
SCORE: (1, 2 or 3)

Your response should be as follows:

SCORE: (1, 2 or 3)
(line break)
JUSTIFICATION: (Without mentioning the student/teacher framing of this prompt, explain why the STUDENT ANSWER is Correct or Incorrect, identify potential sources of bias in the QUESTION, and identify potential sources of bias in the TRUE ANSWER. Use one or two sentences maximum. Keep the answer as concise as possible.)
"""

GRADE_ANSWER_PROMPT_WITH_GRADING_JUSTIFICATION = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)


template = """Please act as an impartial judge and evaluate the quality of the provided answer which attempts to answer the provided question based on a provided context.

  You'll be given a function grading_function which you'll call for each provided context, question and answer to submit your reasoning and score for the correctness, comprehensiveness and readability of the answer. 

  Below is your grading rubric: 

- Correctness: If the answer correctly answer the question, below are the details for different scores:

  - Score 0: the answer is completely incorrect, doesn’t mention anything about the question or is completely contrary to the correct answer.

      - For example, when asked “How to terminate a databricks cluster”, the answer is empty string, or content that’s completely irrelevant, or sorry I don’t know the answer.

  - Score 1: the answer provides some relevance to the question and answers one aspect of the question correctly.

      - Example:

          - Question: How to terminate a databricks cluster

          - Answer: Databricks cluster is a cloud-based computing environment that allows users to process big data and run distributed data processing tasks efficiently.

          - Or answer:  In the Databricks workspace, navigate to the "Clusters" tab. And then this is a hard question that I need to think more about it

  - Score 2: the answer mostly answer the question but is missing or hallucinating on one critical aspect.

      - Example:

          - Question: How to terminate a databricks cluster”

          - Answer: “In the Databricks workspace, navigate to the "Clusters" tab.

          Find the cluster you want to terminate from the list of active clusters.

          And then you’ll find a button to terminate all clusters at once”

  - Score 3: the answer correctly answer the question and not missing any major aspect

      - Example:

          - Question: How to terminate a databricks cluster

          - Answer: In the Databricks workspace, navigate to the "Clusters" tab.

          Find the cluster you want to terminate from the list of active clusters.

          Click on the down-arrow next to the cluster name to open the cluster details.

          Click on the "Terminate" button. A confirmation dialog will appear. Click "Terminate" again to confirm the action.”

- Comprehensiveness: How comprehensive is the answer, does it fully answer all aspects of the question and provide comprehensive explanation and other necessary information. Below are the details for different scores:

  - Score 0: typically if the answer is completely incorrect, then the comprehensiveness is also zero score.

  - Score 1: if the answer is correct but too short to fully answer the question, then we can give score 1 for comprehensiveness.

      - Example:

          - Question: How to use databricks API to create a cluster?

          - Answer: First, you will need a Databricks access token with the appropriate permissions. You can generate this token through the Databricks UI under the 'User Settings' option. And then (the rest is missing)

  - Score 2: the answer is correct and roughly answer the main aspects of the question, but it’s missing description about details. Or is completely missing details about one minor aspect.

      - Example:

          - Question: How to use databricks API to create a cluster?

          - Answer: You will need a Databricks access token with the appropriate permissions. Then you’ll need to set up the request URL, then you can make the HTTP Request. Then you can handle the request response.

      - Example:

          - Question: How to use databricks API to create a cluster?

          - Answer: You will need a Databricks access token with the appropriate permissions. Then you’ll need to set up the request URL, then you can make the HTTP Request. Then you can handle the request response.

  - Score 3: the answer is correct, and covers all the main aspects of the question

- Readability: How readable is the answer, does it have redundant information or incomplete information that hurts the readability of the answer.

  - Score 0: the answer is completely unreadable, e.g. fully of symbols that’s hard to read; e.g. keeps repeating the words that it’s very hard to understand the meaning of the paragraph. No meaningful information can be extracted from the answer.

  - Score 1: the answer is slightly readable, there are irrelevant symbols or repeated words, but it can roughly form a meaningful sentence that cover some aspects of the answer.

      - Example:

          - Question: How to use databricks API to create a cluster?

          - Answer: You you  you  you  you  you  will need a Databricks access token with the appropriate permissions. And then then you’ll need to set up the request URL, then you can make the HTTP Request. Then Then Then Then Then Then Then Then Then

  - Score 2: the answer is correct and mostly readable, but there is one obvious piece that’s affecting the readability (mentioning of irrelevant pieces, repeated words)

      - Example:

          - Question: How to terminate a databricks cluster

          - Answer: In the Databricks workspace, navigate to the "Clusters" tab.

          Find the cluster you want to terminate from the list of active clusters.

          Click on the down-arrow next to the cluster name to open the cluster details.

          Click on the "Terminate" button…………………………………..

          A confirmation dialog will appear. Click "Terminate" again to confirm the action.

  - Score 3: the answer is correct and reader friendly, no obvious piece that affect readability.

- Then final rating:

    - Ratio: 60% correctness + 20% comprehensiveness + 20% readability

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is Incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
SCORE: (1, 2 or 3)

Your response should be as follows:

SCORE: (1, 2 or 3)
"""

GRADE_ANSWER_PROMPT_WITH_GRADING = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)
