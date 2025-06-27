# api/train_api.py
from fastapi import APIRouter
from utils.train_utils import run_training

router = APIRouter()

@router.post("/train")
async def train_endpoint():
    run_training()
    return {"status": "Training started"}
