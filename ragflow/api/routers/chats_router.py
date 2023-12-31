from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from starlette import status
from pydantic import BaseModel, Field

from ragflow.utils.hyperparam_chats import AsyncCallbackHandler
from ragflow.utils.hyperparam_chats import query_chat, create_gen, get_docs

router = APIRouter(
    prefix="/chats",
    tags=["Endpoints to chat with parameterized RAGs"],
)


class ChatQueryRequest(BaseModel):
    hp_id: int = Field(ge=0, description="Hyperparameter run id")
    hyperparameters_results_path: str = (
        Field(min_length=3, description="path to list of hp results"),
    )
    user_id: str = Field(description="user id from db")
    api_keys: dict[str, str] = Field(description="Dictionary of API keys.")
    query: str = Field(min_length=5, description="User query")

    class Config:
        json_schema_extra = {
            "example": {
                "id": 0,
                "hyperparameters_results_path": "./tmp/hyperparameters_results.json",
                "user_id": "1",
                "api_keys": {
                    "OPENAI_API_KEY": "your_api_key_here",
                    "ANOTHER_API_KEY": "another_key_here",
                },
                "query": "My query",
            }
        }


@router.post("/query", status_code=status.HTTP_200_OK)
async def query_chat_model_normal(request: ChatQueryRequest):
    try:
        return query_chat(**request.model_dump())
    except Exception as ex:
        raise HTTPException(status_code=400, detail=str(ex))


@router.post("/get_docs", status_code=status.HTTP_200_OK)
async def query_chat_model_normal(request: ChatQueryRequest):
    try:
        return await get_docs(**request.model_dump())
    except Exception as ex:
        raise HTTPException(status_code=400, detail=str(ex))


@router.post("/query_stream", status_code=status.HTTP_200_OK)
async def query_chat_model_stream(request: ChatQueryRequest):
    stream_it = AsyncCallbackHandler()
    gen = create_gen(**request.model_dump(), stream_it=stream_it)
    return StreamingResponse(gen, media_type="text/event-stream")
