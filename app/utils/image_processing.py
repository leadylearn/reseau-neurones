from typing import Dict, Any
import cv2
import numpy as np
from PIL import Image
import io

def load_image(image_path: str) -> Dict[str, Any]:
    """Charge une image depuis un fichier"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image depuis {image_path}")
        
        # Convertir de BGR à RGB (format standard pour la plupart des modèles)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return {
            "image": image,
            "original_shape": image.shape,
            "dtype": str(image.dtype)
        }
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement de l'image: {str(e)}")

def resize_image(data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Redimensionne une image"""
    try:
        image = data["image"]
        width = params.get("width", 224)
        height = params.get("height", 224)
        
        # Conserver les métadonnées existantes
        metadata = data.get("metadata", {})
        
        # Redimensionner l'image
        resized = cv2.resize(
            image, 
            (width, height),
            interpolation=cv2.INTER_AREA
        )
        
        # Mettre à jour les métadonnées
        metadata.update({
            "resized_shape": resized.shape,
            "original_shape": data.get("original_shape", image.shape)
        })
        
        return {
            "image": resized,
            "metadata": metadata,
            **{k: v for k, v in data.items() if k not in ["image", "metadata"]}
        }
    except Exception as e:
        raise ValueError(f"Erreur lors du redimensionnement: {str(e)}")

def normalize_image(data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise les valeurs des pixels d'une image"""
    try:
        image = data["image"]
        min_val = params.get("min", 0.0)
        max_val = params.get("max", 1.0)
        
        # Convertir en float32 pour la normalisation
        normalized = image.astype(np.float32)
        
        # Normaliser entre 0 et 1
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
        
        # Mettre à l'échelle selon les paramètres
        if min_val != 0.0 or max_val != 1.0:
            normalized = normalized * (max_val - min_val) + min_val
        
        # Mettre à jour les métadonnées
        metadata = data.get("metadata", {})
        metadata["normalization"] = {
            "min": float(normalized.min()),
            "max": float(normalized.max()),
            "dtype": str(normalized.dtype)
        }
        
        return {
            "image": normalized,
            "metadata": metadata,
            **{k: v for k, v in data.items() if k not in ["image", "metadata"]}
        }
    except Exception as e:
        raise ValueError(f"Erreur lors de la normalisation: {str(e)}")

def convert_to_grayscale(data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Convertit une image en niveaux de gris"""
    try:
        image = data["image"]
        
        # Convertir en niveaux de gris si ce n'est pas déjà le cas
        if len(image.shape) == 3 and image.shape[2] == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Ajouter une dimension pour la compatibilité avec les modèles qui s'attendent à 3 canaux
            grayscale = np.expand_dims(grayscale, axis=-1)
            grayscale = np.repeat(grayscale, 3, axis=-1)
        else:
            grayscale = image
        
        # Mettre à jour les métadonnées
        metadata = data.get("metadata", {})
        metadata["grayscale"] = True
        
        return {
            "image": grayscale,
            "metadata": metadata,
            **{k: v for k, v in data.items() if k not in ["image", "metadata"]}
        }
    except Exception as e:
        raise ValueError(f"Erreur lors de la conversion en niveaux de gris: {str(e)}")

def save_image(data: Dict[str, Any], output_path: str) -> None:
    """Sauvegarde une image sur le disque"""
    try:
        image = data["image"]
        
        # Convertir en uint8 si nécessaire
        if image.dtype != np.uint8:
            # Si l'image est normalisée entre 0 et 1, la ramener à 0-255
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convertir de RGB à BGR pour OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(output_path, image)
        return output_path
    except Exception as e:
        raise ValueError(f"Erreur lors de la sauvegarde de l'image: {str(e)}")
