from fastapi import APIRouter
from app.endpoints.server_check import router as server_check_router
from app.endpoints.deepl import router as deepl_router
from app.endpoints.pregunta import router as pregunta_router

api_router = APIRouter()

api_router.include_router(server_check_router, prefix="/server_check", tags=["server_check"])
api_router.include_router(deepl_router, prefix="/deepl", tags=["deepl"])
api_router.include_router(pregunta_router, prefix="/pregunta", tags=["pregunta"])
