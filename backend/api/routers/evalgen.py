from fastapi import APIRouter, HTTPException
from starlette import status
from pydantic import BaseModel, Field

from backend.testsetgen import agenerate_evaluation_set

router = APIRouter(
    prefix="/evalsetgeneration",
    tags=["QA evaluation set generation"],
)


class EvalsetGenerationRequest(BaseModel):
    document_store_path: str = Field(min_length=3, description="path to documents")
    qa_gen_params_path: str = Field(
        min_length=3, description="path to configurations for qa generation"
    )
    eval_dataset_path: str = Field(
        min_length=3,
        description="path to where the generated qa pairs should be stored.",
    )
    user_id: str = Field(
        min_length=1, description="user id, e.g. used to access cached embeddings."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document_store_path": "./tmp/document_store/",  # path to documents
                "qa_gen_params_path": "./tmp/qa_gen_params.json",  # path to list of hyperparameters
                "eval_dataset_path": "./tmp/eval_data.json",  # path to generated evaluation dataset
                "user_id": "3e6e131c-d9f3-4085-a412-f5f8875e34f0",  # user id
            }
        }


@router.post("/start", status_code=status.HTTP_200_OK)
async def start_evalset_generation(gen_request: EvalsetGenerationRequest):
    try:
        await agenerate_evaluation_set(**gen_request.model_dump())
    except Exception as ex:
        print(ex)
        raise HTTPException(status_code=400, detail=str(ex))
