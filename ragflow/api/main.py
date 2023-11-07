import os
from fastapi import FastAPI

from ragflow.api.routers import configs, hp_eval, label_gen

app = FastAPI(
    title="FastAPI RAGflow Documentation",
    version="0.01",
    description="""API for the main component of RAGflow to generate evaluation datasets of QA pairs and to run hyperparameter evaluations.
    """,
    # root path should only be "/api" in production
    root_path="" if os.environ.get("EXECUTION_CONTEXT") in ["DEV", "TEST"] else "/api",
)

app.include_router(configs.router)
app.include_router(label_gen.router)
app.include_router(hp_eval.router)
