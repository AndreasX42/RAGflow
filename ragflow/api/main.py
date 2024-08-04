import os
from fastapi import FastAPI

from ragflow.api.routers import (
    configs_router,
    evals_router,
    gens_router,
    chats_router,
    auth_router,
    user_router,
)

from ragflow.api.database import Base, engine

app = FastAPI(
    title="FastAPI RAGflow Documentation",
    version="0.01",
    description="""API for the main component of RAGflow to generate evaluation datasets of QA pairs and to run hyperparameter evaluations.
    """,
    # root path should only be "/api" in production
    root_path="" if os.environ.get("EXECUTION_CONTEXT") in ["DEV", "TEST"] else "/api",
)

app.include_router(configs_router.router)
app.include_router(gens_router.router)
app.include_router(evals_router.router)
app.include_router(chats_router.router)
app.include_router(auth_router.router)
app.include_router(user_router.router)

Base.metadata.create_all(bind=engine)
