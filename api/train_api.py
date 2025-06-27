from fastapi import APIRouter
from utils.train_utils import run_training

router = APIRouter()

@router.post("/api/train")
def train_models():
    result = run_training()
    return result
