from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from starlette import status
from pydantic import BaseModel, Field
from uuid import UUID

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
    user_id: UUID = Field(description="user id, e.g. used to access cached embeddings")
    api_keys: dict[str, str] = Field(description="Dictionary of API keys.")
    query: str = Field(min_length=5, description="User query")

    class Config:
        json_schema_extra = {
            "example": {
                "id": 0,
                "hyperparameters_results_path": "./tmp/hyperparameters_results.json",
                "user_id": "3e6e131c-d9f3-4085-a412-f5f8875e34f0",
                "api_keys": {
                    "OPENAI_API_KEY": "your_api_key_here",
                    "ANOTHER_API_KEY": "another_key_here",
                },
                "query": "My query",
            }
        }


@router.post("/query", status_code=status.HTTP_200_OK)
async def query_chat_model_normal(request: ChatQueryRequest):
    args = request.model_dump()
    args["user_id"] = str(args["user_id"])
    try:
        return query_chat(**args)
    except Exception as ex:
        raise HTTPException(status_code=400, detail=str(ex))


@router.post("/get_docs", status_code=status.HTTP_200_OK)
async def query_chat_model_normal(request: ChatQueryRequest):
    args = request.model_dump()
    args["user_id"] = str(args["user_id"])
    try:
        return await get_docs(**args)
    except Exception as ex:
        raise HTTPException(status_code=400, detail=str(ex))


@router.post("/query_stream", status_code=status.HTTP_200_OK)
async def query_chat_model_stream(request: ChatQueryRequest):
    args = request.model_dump()
    args["user_id"] = str(args["user_id"])

    stream_it = AsyncCallbackHandler()
    gen = create_gen(**args, stream_it=stream_it)
    return StreamingResponse(gen, media_type="text/event-stream")
