from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import json
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Neural Network Simulation API",
    description="API pour la simulation de réseaux de neurones 2D",
    version="0.1.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles Pydantic
class TrainingData(BaseModel):
    input_data: List[float]
    target: List[float]

class TrainingRequest(BaseModel):
    model_id: str
    training_data: List[TrainingData]
    epochs: int = 100
    learning_rate: float = 0.01

# Routes
@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de simulation de réseaux de neurones 2D"}

@app.post("/train/")
async def train_model(request: TrainingRequest):
    """
    Endpoint pour l'entraînement du modèle
    """
    try:
        # Ici viendra la logique d'entraînement
        logger.info(f"Début de l'entraînement pour le modèle {request.model_id}")
        
        # Simulation de l'entraînement
        training_metrics = {
            "model_id": request.model_id,
            "epochs_completed": request.epochs,
            "final_loss": 0.01,  # Valeur simulée
            "accuracy": 0.98     # Valeur simulée
        }
        
        return {
            "status": "success",
            "message": "Entraînement terminé avec succès",
            "metrics": training_metrics
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """
    Récupère les informations d'un modèle
    """
    # Ici, vous récupéreriez les informations du modèle depuis votre base de données
    return {
        "model_id": model_id,
        "status": "trained",
        "created_at": "2025-12-17T23:00:00Z"
    }

from fastapi import UploadFile, File, HTTPException
import os
from datetime import datetime
from fastapi.responses import JSONResponse

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    """
    Traite une image téléchargée via le pipeline de traitement
    """
    try:
        # Vérifier que le fichier est une image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")
        
        # Créer un nom de fichier unique avec un timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.filename)[1]
        filename = f"{timestamp}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Sauvegarder le fichier temporairement
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Ici, vous pouvez ajouter votre logique de traitement d'image
        # Par exemple, appeler votre fonction de traitement d'image
        # result = await process_image_pipeline(file_path)
        
        # Pour l'instant, on retourne simplement une réponse de succès
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Image traitée avec succès",
                "data": {
                    "original_filename": file.filename,
                    "saved_path": file_path,
                    "processed_image_url": f"/processed/{filename}",
                    "processing_steps": ["resize", "normalize", "validate"]
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de l'image: {str(e)}")

from fastapi.staticfiles import StaticFiles

# Servir les images traitées
app.mount("/processed", StaticFiles(directory=UPLOAD_DIR), name="processed")
