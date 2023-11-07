from fastapi import APIRouter, HTTPException
from starlette import status
from pydantic import BaseModel, Field
from uuid import UUID

from ragflow.generation import agenerate_evaluation_set

router = APIRouter(
    tags=["Evaluation label set generation"],
)


class GenerationRequest(BaseModel):
    document_store_path: str = Field(min_length=3, description="path to documents")
    label_dataset_gen_params_path: str = Field(
        min_length=3, description="path to configurations for qa generation"
    )
    label_dataset_path: str = Field(
        min_length=3,
        description="path to where the generated qa pairs should be stored.",
    )
    user_id: UUID = Field(description="user id, e.g. used to access cached embeddings.")
    api_keys: dict[str, str] = Field(description="Dictionary of API keys.")

    class Config:
        json_schema_extra = {
            "example": {
                "document_store_path": "./tmp/document_store/",  # path to documents
                "label_dataset_gen_params_path": "./tmp/label_dataset_gen_params.json",  # path to list of hyperparameters
                "label_dataset_path": "./tmp/label_dataset.json",  # path to generated evaluation dataset
                "user_id": "3e6e131c-d9f3-4085-a412-f5f8875e34f0",  # user id
                "api_keys": {
                    "OPENAI_API_KEY": "your_api_key_here",
                    "ANOTHER_API_KEY": "another_key_here",
                },
            }
        }


@router.post("/generation", status_code=status.HTTP_200_OK)
async def start_evalset_generation(gen_request: GenerationRequest):
    args = gen_request.model_dump()
    args["user_id"] = str(args["user_id"])

    try:
        await agenerate_evaluation_set(**args)
    except Exception as ex:
        print(ex)
        raise HTTPException(status_code=400, detail=str(ex))
