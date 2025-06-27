# api/train_api.py
from fastapi import APIRouter
from utils.train_utils import run_training

router = APIRouter()

@router.post("/train")
async def train_endpoint(csv_path: str = "training_data.csv"):
    try:
        # Eğitim başlatılıyor ve metrik sonuçları alınıyor
        results = run_training(csv_path=csv_path)
        
        return {
            "status": "✅ Training completed successfully",
            "metrics": results
        }

    except Exception as e:
        return {
            "status": "❌ Training failed",
            "error": str(e)
        }
