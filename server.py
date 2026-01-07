from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dossiers
UPLOAD_FOLDER = "uploaded_images"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        # Sauvegarder le fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ext = Path(file.filename).suffix
        file_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}{file_ext}")
        
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Traitement de l'image
        img = cv2.imread(file_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Impossible de charger l'image")
        
        # Liste des étapes de traitement
        processing_steps = ["Chargement de l'image"]
        
        # 1. Redimensionnement à 255x255
        img_resized = cv2.resize(img, (50, 50))
        processing_steps.append("Redimensionnement à 50x50 pixels")
        
        # 2. Conversion en niveaux de gris
        processed_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        processing_steps.append("Conversion en niveaux de gris")
        
        # Sauvegarder l'image traitée
        output_path = os.path.join(OUTPUT_FOLDER, f"processed_{timestamp}{file_ext}")
        cv2.imwrite(output_path, processed_img)
        
        return {
            "success": True,
            "message": "Image traitée avec succès",
            "data": {
                "original_path": file_path,
                "processed_path": output_path,
                "processing_steps": processing_steps,
                "prediction": 1
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)