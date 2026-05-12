from fastapi import FastAPI

app = FastAPI(
    title="Signal Conso API",
    version="0.1.0",
    description=(
        "API Signal Conso — prédictions ML, gestion des tickets "
        "et orchestration des flows Prefect KPI."
    ),
)


@app.get("/", tags=["root"])
def root() -> dict:
    return {"message": "Bienvenue sur la plateforme intelligente de Signal Conso !"}
