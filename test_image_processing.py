import os
import asyncio
import cv2
import numpy as np
from typing import Union, Dict, Any, List, Optional
import json
from datetime import datetime, timezone
from pathlib import Path

# Import des fonctions de traitement d'image
def save_image(image_data, output_path):
    """
    Sauvegarde une image sur le disque
    
    Args:
        image_data: Données de l'image (numpy array ou dictionnaire contenant 'image_data')
        output_path: Chemin de sortie pour l'image
    """
    try:
        print(f"Tentative de sauvegarde de l'image vers: {os.path.abspath(output_path)}")
        print(f"Type des données d'image: {type(image_data)}")
        
        # Si les données sont dans un dictionnaire
        if isinstance(image_data, dict):
            if 'image_data' in image_data:
                image_data = image_data['image_data']
            elif 'data' in image_data:
                image_data = image_data['data']
        
        # Vérifier que les données sont un tableau numpy
        if not isinstance(image_data, np.ndarray):
            print(f"Conversion des données en tableau numpy. Type actuel: {type(image_data)}")
            if hasattr(image_data, 'numpy'):  # Pour les tenseurs PyTorch
                image_data = image_data.numpy()
            else:
                image_data = np.array(image_data)
        
        # S'assurer que l'image est au format uint8 (0-255)
        if image_data.dtype != np.uint8:
            if image_data.max() <= 1.0:  # Si les valeurs sont normalisées entre 0 et 1
                image_data = (image_data * 255).astype(np.uint8)
            else:
                image_data = image_data.astype(np.uint8)
        
        # Si l'image est en niveaux de gris mais a 3 canaux, on la convertit en un seul canal
        if len(image_data.shape) == 3 and image_data.shape[2] == 1:
            image_data = image_data.reshape(image_data.shape[:2])
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Tentative d'écriture de l'image. Taille: {image_data.shape}, Type: {image_data.dtype}")
        
        # Sauvegarder l'image
        success = cv2.imwrite(output_path, image_data)
        if not success:
            raise Exception(f"Échec de l'écriture du fichier {output_path}")
            
        print(f"Image sauvegardée avec succès : {os.path.abspath(output_path)}")
        return output_path
        
    except Exception as e:
        error_msg = f"Erreur lors de la sauvegarde de l'image: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        raise Exception(error_msg)

class ProcessingResult:
    def __init__(self, input_data: Union[str, Dict], data_type: str):
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.data_type = data_type  # 'image' ou 'text'
        self.original_data = input_data
        self.processed_data = None
        self.metadata = {}
        self.processing_steps = []
        
    def add_processing_step(self, step_name: str, params: Dict = None):
        self.processing_steps.append({
            "step": step_name,
            "parameters": params or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        def serialize(obj):
            if hasattr(obj, 'isoformat'):  # Pour les objets datetime
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize(item) for item in obj]
            return str(obj)  # Convertit les autres objets non sérialisables en chaîne

        data = {
            "timestamp": self.timestamp,
            "data_type": self.data_type,
            "original_data": self.original_data,
            "processed_data": self.processed_data,
            "metadata": self.metadata,
            "processing_steps": self.processing_steps
        }
        
        # Utiliser json.dumps avec default=serialize pour gérer les types non standards
        import json
        return json.loads(json.dumps(data, default=serialize))
    
    def save_to_json(self, output_dir: str) -> str:
        """Sauvegarde les résultats dans un fichier JSON"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processing_result_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        return filepath

# Fonction pour traiter le texte
def process_text(text: str, steps: list) -> Dict:
    """Traite un texte avec les étapes spécifiées"""
    result = text
    processing_steps = []
    
    for step in steps:
        if step["name"] == "uppercase":
            result = result.upper()
        elif step["name"] == "lowercase":
            result = result.lower()
        elif step["name"] == "remove_punctuation":
            import string
            result = result.translate(str.maketrans('', '', string.punctuation))
        
        processing_steps.append({
            "step": step["name"],
            "result": result
        })
    
    return {
        "processed_text": result,
        "processing_steps": processing_steps
    }

# Fonction utilitaire pour vérifier l'existence d'un fichier
def check_file_exists(file_path: str) -> bool:
    """Vérifie si un fichier existe et est accessible"""
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier '{file_path}' n'existe pas.")
        return False
    return True

# Mise à jour de la fonction principale
async def process_data(input_data: Union[str, Dict], data_type: str, output_dir: str = "output") -> Optional[ProcessingResult]:
    """
    Traite les données (image ou texte) et sauvegarde les résultats
    
    Args:
        input_data: Chemin de l'image ou texte à traiter
        data_type: 'image' ou 'text'
        output_dir: Dossier de sortie pour les résultats
    
    Returns:
        ProcessingResult: L'objet résultat du traitement, ou None en cas d'erreur
    """
    # Vérification des entrées
    if data_type == "image" and not check_file_exists(input_data):
        return None

    # Initialisation du résultat
    result = ProcessingResult(input_data, data_type)
    
    if data_type == "image":
        # Traitement d'image
        from app.services.processing import ProcessingService
        from app.schemas.processing import ProcessingRequest, ProcessingPipelineConfig, DataType, ProcessingStep
        
        service = ProcessingService()
        
        # Configuration du pipeline de traitement d'image
        pipeline_config = ProcessingPipelineConfig(
            pipeline_id="image_processing",
            name="Traitement d'images",
            description="Pipeline de traitement d'images de base",
            input_type=DataType.IMAGE,
            output_type=DataType.IMAGE,
            steps=[
                ProcessingStep(
                    name="resize", 
                    description="Redimensionnement de l'image", 
                    parameters={"width": 800, "height": 600},
                    order=1
                ),
                ProcessingStep(
                    name="grayscale", 
                    description="Conversion en niveaux de gris", 
                    parameters={},
                    order=2
                ),
                ProcessingStep(
                    name="normalize", 
                    description="Normalisation des valeurs de pixels", 
                    parameters={"min": 0, "max": 1},
                    order=3
                )
            ]
        )
        
        # Enregistrement du pipeline
        service.register_pipeline(pipeline_config)
        
        # Configuration des paramètres de traitement
        processing_params = {
            "resize": {"width": 800, "height": 600},
            "grayscale": {"enabled": True},
            "normalize": {"min": 0, "max": 1}
        }
        
        # Chargement de l'image pour vérification
        img = cv2.imread(input_data)
        if img is None:
            result.metadata["error"] = f"Impossible de charger l'image: {input_data}"
            print(f"Erreur: {result.metadata['error']}")
            return result
            
        # Enregistrement des métadonnées
        result.metadata.update({
            "pipeline_config": pipeline_config.model_dump(),
            "processing_steps": processing_params,
            "original_size": f"{img.shape[1]}x{img.shape[0]}",
            "channels": img.shape[2] if len(img.shape) > 2 else 1
        })
        
        # Ajout des étapes de traitement au pipeline
        pipeline = service.get_pipeline("image_processing")
        
        # Fonction utilitaire pour charger une image
        def load_image(image_path):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
            return img
            
        # Fonction de redimensionnement
        def resize_image(img_data, params):
            if isinstance(img_data, dict) and 'image_path' in img_data:
                # Si on reçoit un chemin, on charge d'abord l'image
                img = load_image(img_data['image_path'])
            elif isinstance(img_data, np.ndarray):
                # Si on reçoit déjà un tableau numpy
                img = img_data
            else:
                raise ValueError("Format d'image non supporté. Attendu: chemin d'accès ou tableau numpy")
                
            return cv2.resize(img, (params["width"], params["height"]))
            
        # Redimensionnement
        if "resize" in processing_params:
            params = processing_params["resize"]
            pipeline.add_step("resize", 
                lambda img_data, p=params: resize_image(img_data, p))
        
        # Conversion en niveaux de gris
        def convert_to_grayscale(img_data, _):
            if isinstance(img_data, dict) and 'image_path' in img_data:
                img = load_image(img_data['image_path'])
            elif isinstance(img_data, np.ndarray):
                img = img_data
            else:
                raise ValueError("Format d'image non supporté")
                
            if len(img.shape) == 3:  # Si l'image est en couleur
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img  # Déjà en niveaux de gris
            
        if "grayscale" in processing_params and processing_params["grayscale"].get("enabled", False):
            pipeline.add_step("grayscale", convert_to_grayscale)
        
        # Normalisation
        def normalize_image(img_data, params):
            if isinstance(img_data, dict) and 'image_path' in img_data:
                img = load_image(img_data['image_path'])
            elif isinstance(img_data, np.ndarray):
                img = img_data.copy()
            else:
                raise ValueError("Format d'image non supporté")
            
            # Conversion en float32 pour la normalisation
            img_float = img.astype(np.float32)
            
            # Normalisation entre 0 et 1
            img_normalized = cv2.normalize(
                img_float, None, 
                alpha=0,  # Valeur minimale après normalisation
                beta=1.0,  # Valeur maximale après normalisation
                norm_type=cv2.NORM_MINMAX
            )
            
            # Conversion en uint8 (0-255) pour l'affichage
            return (img_normalized * 255).astype(np.uint8)
            
        if "normalize" in processing_params:
            params = processing_params["normalize"]
            pipeline.add_step("normalize", 
                lambda img_data, p=params: normalize_image(img_data, p))
        
        # Création de la requête de traitement
        request = ProcessingRequest(
            pipeline_id="image_processing",
            input_data={"image_path": input_data},
            parameters={"steps": processing_params}
        )
        
        try:
            response = await service.process(request)
            if response.status == "completed":
                # Créer le dossier output s'il n'existe pas
                os.makedirs('output', exist_ok=True)
                
                # Chemin de sortie dans le dossier output
                output_image_path = os.path.join('output', f"processed_{os.path.basename(input_data)}")
                
                # S'assure que le résultat est un dictionnaire
                processed_image = response.result
                if not isinstance(processed_image, dict):
                    processed_image = {"image_data": processed_image}
                
                # Sauvegarde l'image
                save_image(processed_image.get("image_data", processed_image), output_image_path)
                print(f"Image traitée enregistrée dans : {os.path.abspath(output_image_path)}")
                
                # Mise à jour des résultats
                result.processed_data = {
                    "output_path": output_image_path,
                    "processing_steps": [
                        {"step": step.name, "parameters": step.parameters} 
                        for step in pipeline_config.steps
                    ]
                }
                
                # Affichage des images avant/après si l'option --no-display n'est pas activée
                display_images(input_data, output_image_path, no_display=args.no_display)
                
        except Exception as e:
            result.metadata["error"] = str(e)
    
    elif data_type == "text":
        # Traitement de texte
        processing_steps = [
            {"name": "uppercase", "enabled": True},
            {"name": "remove_punctuation", "enabled": True}
        ]
        
        # Traitement
        processed = process_text(input_data, processing_steps)
        
        # Mise à jour des résultats
        result.processed_data = {
            "processed_text": processed["processed_text"],
            "processing_steps": processed["processing_steps"]
        }
        
        # Affichage du texte avant/après
        print("\n=== Texte original ===")
        print(input_data)
        print("\n=== Texte traité ===")
        print(processed["processed_text"])
    
    # Créer le dossier resultats s'il n'existe pas
    os.makedirs('resultats', exist_ok=True)
    
    # Sauvegarde des résultats au format JSON dans le dossier resultats
    json_path = result.save_to_json('resultats')
    print(f"\nRésultats JSON sauvegardés dans : {os.path.abspath(json_path)}")
    print(f"Image traitée sauvegardée dans : {os.path.abspath(os.path.join('output', f'processed_{os.path.basename(input_data)}'))}")
    
    return result

# Fonction utilitaire pour afficher les images avant/après
def display_images(original_path: str, processed_path: str, no_display=False):
    """Affiche côte à côte l'image originale et l'image traitée"""
    if no_display:
        print("Affichage désactivé (--no-display activé)")
        return
        
    import matplotlib.pyplot as plt
    
    try:
        original = cv2.imread(original_path)
        if original is None:
            print(f"Erreur: Impossible de charger l'image originale: {original_path}")
            return
            
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        processed = cv2.imread(processed_path)
        if processed is None:
            print(f"Erreur: Impossible de charger l'image traitée: {processed_path}")
            return
            
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title("Image originale")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(processed)
        plt.title("Image traitée")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Erreur lors de l'affichage des images: {e}")
        import traceback
        traceback.print_exc()

# Exemple d'utilisation
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Traitement d'images ou de texte")
    parser.add_argument("--input", required=True, help="Chemin de l'image ou texte à traiter")
    parser.add_argument("--type", required=True, choices=["image", "text"], 
                       help="Type de données à traiter")
    parser.add_argument("--output", default="output", help="Dossier de sortie")
    parser.add_argument("--no-display", action="store_true", help="Ne pas afficher les graphiques")
    
    args = parser.parse_args()
    
    # Création des dossiers de sortie s'ils n'existent pas
    os.makedirs('output', exist_ok=True)
    os.makedirs('resultats', exist_ok=True)
    
    # Exécution du traitement avec le dossier de sortie spécifié
    asyncio.run(process_data(args.input, args.type, args.output))