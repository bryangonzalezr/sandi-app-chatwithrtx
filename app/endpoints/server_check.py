from fastapi import APIRouter, Depends, HTTPException



router = APIRouter()

@router.get("/test")
async def test():
    return {"message": "Hello, World!"}