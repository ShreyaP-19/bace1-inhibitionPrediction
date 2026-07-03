import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rdkit import Chem
import sys
from contextlib import asynccontextmanager

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main.predict import load_model, predict_with_model

# --- Global Variables ---
model_data = {}

# --- Lifespan Manager for Model Loading ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_data
    # Robust path: relative to THIS file (backend/api.py) -> up one level -> bace_model.pth
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bace_model.pth"))
    
    try:
        # Check if file exists first to avoid confusing torch errors
        if os.path.exists(model_path):
             print(f"INFO: Loading model from {model_path}...")
             model, tokenizer, device = load_model(model_path)
             model_data["model"] = model
             model_data["tokenizer"] = tokenizer
             model_data["device"] = device
             print(f"INFO: Model loaded successfully!")
        else:
             print(f"WARNING: Model file not found at {model_path}. Running in MOCK mode.")
             model_data = {}
            
    except Exception as e:
        print(f"WARNING: Failed to load model: {e}")
        model_data = {}
        
    yield
    # Clean up resources if needed
    model_data.clear()

# --- App Initialization ---
app = FastAPI(title="BACE-1 Prediction API", lifespan=lifespan)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class MoleculeInput(BaseModel):
    smiles: str

class PredictionResponse(BaseModel):
    isInhibitor: bool
    predictedIC50: float
    confidenceScore: float

# --- Routes ---
@app.get("/")
def read_root():
    status = "Active" if model_data else "Mock Mode (Model Not Loaded)"
    return {"message": "BACE-1 Inhibitor Prediction API is running", "model_status": status}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: MoleculeInput):
    # 1. Scientific Validation
    if not data.smiles:
         raise HTTPException(status_code=400, detail="SMILES string cannot be empty")
         
    mol = Chem.MolFromSmiles(data.smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")

    # 2. Prediction Logic
    if "model" in model_data:
        try:
            prob, pic50 = predict_with_model(
                data.smiles, 
                model_data["model"], 
                model_data["tokenizer"], 
                model_data["device"]
            )
            
            if prob is None:
                raise ValueError("Prediction returned None")
                
            return {
                "isInhibitor": prob > 0.5,
                "predictedIC50": round(pic50, 4),
                "confidenceScore": round(prob, 4)
            }
            
        except Exception as e:
             print(f"Inference Error: {e}")
             raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    else:
        # Fallback to Mock Mode (if model failed to load)
        import random
        return {
            "isInhibitor": random.random() > 0.5,
            "predictedIC50": round(random.uniform(4.0, 9.0), 2),
            "confidenceScore": round(random.random(), 2)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
