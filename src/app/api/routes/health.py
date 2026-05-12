from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    return {"status": "OK", "service": "plateforme intelligente de Signal Conso !"}
