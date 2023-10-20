from fastapi import FastAPI

from backend.api.routers import configs, hp_eval, label_gen

app = FastAPI(
    title="FastAPI RAGflow Backend Documentation",
    version="0.01",
    description="""API for the backend component of RAGflow to generate evaluation datasets of QA pairs and to run hyperparameter evaluations.
    """,
    root_path="/api",
)

app.include_router(configs.router)
app.include_router(label_gen.router)
app.include_router(hp_eval.router)
