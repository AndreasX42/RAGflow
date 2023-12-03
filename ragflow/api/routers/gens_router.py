from fastapi import APIRouter, HTTPException
from starlette import status
from pydantic import BaseModel, Field

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
    user_id: int = Field(ge=1, description="user id from db")
    api_keys: dict[str, str] = Field(description="Dictionary of API keys.")

    class Config:
        json_schema_extra = {
            "example": {
                "document_store_path": "./tmp/document_store/",  # path to documents
                "label_dataset_gen_params_path": "./tmp/label_dataset_gen_params.json",  # path to list of hyperparameters
                "label_dataset_path": "./tmp/label_dataset.json",  # path to generated evaluation dataset
                "user_id": "1",  # user id
                "api_keys": {
                    "OPENAI_API_KEY": "your_api_key_here",
                    "ANOTHER_API_KEY": "another_key_here",
                },
            }
        }


@router.post("/generation", status_code=status.HTTP_200_OK)
async def start_evalset_generation(gen_request: GenerationRequest):
    try:
        await agenerate_evaluation_set(**gen_request.model_dump())
    except Exception as ex:
        print(ex)
        raise HTTPException(status_code=400, detail=str(ex))
