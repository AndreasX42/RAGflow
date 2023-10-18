from fastapi import APIRouter, HTTPException
from starlette import status
from pydantic import BaseModel, Field
from uuid import UUID

from backend.evaluation import arun_evaluation


router = APIRouter(
    prefix="/evaluations",
    tags=["Hyperparameter evaluations"],
)


class EvaluationRunRequest(BaseModel):
    document_store_path: str = Field(min_length=3, description="path to documents")
    eval_params_path: str = Field(
        min_length=3, description="path to list of hyperparameters"
    )
    eval_dataset_path: str = Field(
        min_length=3, description="path to generated evaluation dataset"
    )
    eval_results_path: str = Field(
        min_length=3, description="path to list of eval results"
    )
    hp_runs_data_path: str = Field(
        min_length=3, description="path to list of generated"
    )
    user_id: UUID = Field(description="user id, e.g. used to access cached embeddings")
    api_keys: dict[str, str] = Field(description="Dictionary of API keys.")

    class Config:
        json_schema_extra = {
            "example": {
                "document_store_path": "./tmp/document_store/",  # path to documents
                "eval_params_path": "./tmp/eval_params.json",  # path to list of hyperparameters
                "eval_dataset_path": "./tmp/eval_data.json",  # path to generated evaluation dataset
                "eval_results_path": "./tmp/eval_results.json",  # path to list of eval results
                "hp_runs_data_path": "./tmp/hp_runs_data.csv",  # path to list of generated predictions and retrieved docs
                "user_id": "3e6e131c-d9f3-4085-a412-f5f8875e34f0",  # user id
                "api_keys": {
                    "OPENAI_API_KEY": "your_api_key_here",
                    "ANOTHER_API_KEY": "another_key_here",
                },
            }
        }


@router.post("/start", status_code=status.HTTP_200_OK)
async def start_evaluation_run(eval_request: EvaluationRunRequest):
    try:
        await arun_evaluation(**eval_request.model_dump())
    except Exception as ex:
        print(ex)
        raise HTTPException(status_code=400, detail=str(ex))
