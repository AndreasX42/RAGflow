from fastapi import APIRouter, HTTPException
from starlette import status
from pydantic import BaseModel, Field
from uuid import UUID

from ragflow.evaluation import arun_evaluation


router = APIRouter(
    tags=["Hyperparameter evaluation"],
)


class EvaluationRequest(BaseModel):
    document_store_path: str = Field(min_length=3, description="path to documents")
    label_dataset_path: str = Field(
        min_length=3, description="path to generated evaluation dataset"
    )
    hyperparameters_path: str = Field(
        min_length=3, description="path to list of hyperparameters"
    )
    hyperparameters_results_path: str = Field(
        min_length=3, description="path to list of hp results"
    )
    hyperparameters_results_data_path: str = Field(
        min_length=3,
        description="path to list of additional data generated during hp eval",
    )
    user_id: UUID = Field(description="user id, e.g. used to access cached embeddings")
    api_keys: dict[str, str] = Field(description="Dictionary of API keys.")

    class Config:
        json_schema_extra = {
            "example": {
                "document_store_path": "./tmp/document_store/",  # path to documents
                "hyperparameters_path": "./tmp/hyperparameters.json",  # path to list of hyperparameters
                "label_dataset_path": "./tmp/label_dataset.json",  # path to generated evaluation dataset
                "hyperparameters_results_path": "./tmp/hyperparameters_results.json",  # path to list of eval results
                "hyperparameters_results_data_path": "./tmp/hyperparameters_results_data.csv",  # path to list of generated predictions and retrieved docs
                "user_id": "3e6e131c-d9f3-4085-a412-f5f8875e34f0",  # user id
                "api_keys": {
                    "OPENAI_API_KEY": "your_api_key_here",
                    "ANOTHER_API_KEY": "another_key_here",
                },
            }
        }


@router.post("/evaluation", status_code=status.HTTP_200_OK)
async def start_evaluation_run(eval_request: EvaluationRequest):
    args = eval_request.model_dump()
    args["user_id"] = str(args["user_id"])
    try:
        await arun_evaluation(**args)
    except Exception as ex:
        print(ex)
        raise HTTPException(status_code=400, detail=str(ex))
