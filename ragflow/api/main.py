import os
from fastapi import FastAPI

from ragflow.api.routers import configs, evals, gens, chats

app = FastAPI(
    title="FastAPI RAGflow Documentation",
    version="0.01",
    description="""API for the main component of RAGflow to generate evaluation datasets of QA pairs and to run hyperparameter evaluations.
    """,
    # root path should only be "/api" in production
    root_path="" if os.environ.get("EXECUTION_CONTEXT") in ["DEV", "TEST"] else "/api",
)

app.include_router(configs.router)
app.include_router(gens.router)
app.include_router(evals.router)
app.include_router(chats.router)
