from dotenv import load_dotenv
import os
import uvicorn
from app.routes import api_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.aux_fun.modelo import model_check_existence

app = FastAPI()

load_dotenv()

#cors para despues
# origins = os.getenv("CORS_ORIGINS").split(",")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# en caso de que necesitemos hacer algo que requiera ejecutarse al inicio
@app.on_event("startup")
async def startup_event():
    # Initialize database connection
    print(model_check_existence())

app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)