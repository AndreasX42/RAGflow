from fastapi import FastAPI

from backend.api.routers import configs, evals, evalgen

app = FastAPI(
    title="FastAPI Docs",
    version="0.01",
    description="""API for the backend component of RAGflow to generate evaluation sets of QA pairs and to run hyperparameter evaluations.
    """,
)

app.include_router(configs.router)
app.include_router(evals.router)
app.include_router(evalgen.router)
