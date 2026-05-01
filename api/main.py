from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
import os

# Add project root to sys.path so we can import the webapp scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp.scripts.predict_with_model import generate_prediction

app = FastAPI(title="SolarCast API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    date: str
    time: str
    manifestPath: str
    dataRoot: Optional[str] = None
    modelName: Optional[str] = None

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "SolarCast API"}

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        payload = req.model_dump()
        result = generate_prediction(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
