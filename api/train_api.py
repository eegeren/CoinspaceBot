from fastapi import APIRouter
from utils.train_utils import run_training

router = APIRouter()

@router.post("/api/train")
async def train_model():
    run_training()
    return {"status": "Training started"}
