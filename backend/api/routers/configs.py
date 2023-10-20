from fastapi import APIRouter
from starlette import status

from backend.commons.configurations import (
    LLM_MODELS,
    EMB_MODELS,
    CVGradeAnswerPrompt,
    CVGradeRetrieverPrompt,
    CVRetrieverSearchType,
    CVSimilarityMethod,
)

router = APIRouter(
    prefix="/configs",
    tags=["Configurations"],
)


@router.get("/llm_models", status_code=status.HTTP_200_OK)
async def get_list_of_supported_llm_models():
    return LLM_MODELS


@router.get("/embedding_models", status_code=status.HTTP_200_OK)
async def list_of_embedding_models():
    return EMB_MODELS


@router.get("/retriever_similarity_methods", status_code=status.HTTP_200_OK)
async def list_of_similarity_methods_for_retriever():
    return [e.value for e in CVSimilarityMethod]


@router.get("/retriever_search_types", status_code=status.HTTP_200_OK)
async def list_of_search_types_for_retriever():
    return [e.value for e in CVRetrieverSearchType]


@router.get("/grade_answer_prompts", status_code=status.HTTP_200_OK)
async def list_of_prompts_for_grading_answers():
    return [e.value for e in CVGradeAnswerPrompt]


@router.get("/grade_documents_prompts", status_code=status.HTTP_200_OK)
async def list_of_prompts_for_grading_documents_retrieved():
    return [e.value for e in CVGradeRetrieverPrompt]
