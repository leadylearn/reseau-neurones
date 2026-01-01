import io
import json
import base64
import tempfile
import subprocess
import os
from datetime import datetime, timezone
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np

import numpy as np
from PIL import Image
import streamlit as st

from app.utils.compression import DataCompressor
from app.utils.image_processing import (
    resize_image,
    normalize_image,
    convert_to_grayscale,
    save_image,
)
from app.schemas.processing import ProcessingRequest, ProcessingResponse, ProcessingStatus, ProcessingPipelineConfig, ProcessingStep, DataType


st.set_page_config(page_title="Pipeline Demo", page_icon="🧪", layout="wide")

compressor = DataCompressor()


def np_from_pil(img: Image.Image) -> np.ndarray:
    return np.array(img)


def pil_from_np(arr: np.ndarray) -> Image.Image:
    # Si normalisé [0,1], repasser en 0-255 pour affichage
    if arr.dtype != np.uint8:
        amax = float(arr.max()) if arr.size else 1.0
        amin = float(arr.min()) if arr.size else 0.0
        rng = (amax - amin) or 1.0
        arr8 = ((arr - amin) / rng * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr8)
    return Image.fromarray(arr)


def compress_bytes_to_b64(raw: bytes) -> str:
    return compressor.compress_to_base64(raw)


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def process_image_with_pipeline(image_path: str) -> dict:
    """
    Traite une image avec le pipeline de traitement
    
    Args:
        image_path: Chemin vers l'image à traiter
        
    Returns:
        dict: Résultats du traitement
    """
    try:
        # Créer un dossier pour les résultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.abspath(os.path.join('resultats', f'processing_{timestamp}'))
        os.makedirs(output_dir, exist_ok=True)
        
        # Convertir le chemin en chemin absolu
        abs_image_path = os.path.abspath(image_path)
        
        # Vérifier que le fichier source existe
        if not os.path.exists(abs_image_path):
            return {
                'success': False,
                'error': f"Le fichier source n'existe pas: {abs_image_path}"
            }
            
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs('output', exist_ok=True)
        os.makedirs('resultats', exist_ok=True)
        
        # Construire la commande avec des chemins absolus
        cmd = [
            'python', 'test_image_processing.py',
            '--input', abs_image_path,
            '--type', 'image',
            '--output', output_dir,
            '--no-display'
        ]
        
        print(f"Exécution de la commande: {' '.join(cmd)}")
        
        # Exécuter la commande avec le répertoire de travail du projet
        result = subprocess.run(
            cmd, 
            cwd=os.getcwd(),
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        print(f"Sortie du processus:\n{result.stdout}")
        if result.stderr:
            print(f"Erreur du processus:\n{result.stderr}")
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': f"Erreur lors du traitement (code {result.returncode}): {result.stderr}"
            }
            
        # Chercher le fichier de résultats JSON le plus récent
        result_files = list(Path('resultats').glob('*.json'))
        if not result_files:
            return {
                'success': False,
                'error': f'Aucun fichier de résultat trouvé dans {os.path.abspath("resultats")}'
            }
            
        # Trier par date de modification (le plus récent en premier)
        result_files.sort(key=os.path.getmtime, reverse=True)
        latest_result = result_files[0]
        
        # Lire et retourner les résultats
        with open(latest_result, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
            
            # Ajouter le chemin du répertoire de sortie aux résultats
            result_data['output_dir'] = os.path.abspath(output_dir)
            
            return {
                'success': True,
                'data': result_data
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def display_json_schema():
    """Affiche le schéma JSON de l'API"""
    req_schema = ProcessingRequest.model_json_schema()
    resp_schema = ProcessingResponse.model_json_schema()
    with st.expander("Schéma JSON - ProcessingRequest"):
        st.code(json.dumps(req_schema, indent=2, ensure_ascii=False), language="json")
    with st.expander("Schéma JSON - ProcessingResponse"):
        st.code(json.dumps(resp_schema, indent=2, ensure_ascii=False), language="json")


st.title("🧪 Prototype Interface - Pipeline Normalisation + Sérialisation + Compression")
mode = st.radio("Type d'entrée", ["Image", "Texte"], horizontal=True)

if mode == "Image":
    file = st.file_uploader("Importer une image", type=["png", "jpg", "jpeg"])

    colA, colB = st.columns(2)
    with colA:
        resize_w = st.number_input("Largeur", min_value=16, value=256, step=16)
        resize_h = st.number_input("Hauteur", min_value=16, value=256, step=16)
    with colB:
        do_gray = st.checkbox("Niveaux de gris", value=False)
        do_normalize = st.checkbox("Normaliser [0,1]", value=True)

    if file is not None:
        original = Image.open(file).convert("RGB")
        st.subheader("Aperçu original")
        st.image(original, use_column_width=True)

        # Construire un "data" dict compatible avec utilitaires existants
        data = {"image": np_from_pil(original), "metadata": {}}

        # Steps config
        steps_cfg = [
            ProcessingStep(name="resize", description="Redimension", parameters={"width": int(resize_w), "height": int(resize_h)}, order=1, required=True),
            ProcessingStep(name="grayscale", description="Niveaux de gris", parameters={}, order=2, required=False),
            ProcessingStep(name="normalize", description="Normalisation", parameters={"min": 0.0, "max": 1.0}, order=3, required=False),
        ]
        pipe_cfg = ProcessingPipelineConfig(
            pipeline_id="image_processing",
            name="Image Processing",
            description="Resize/Grayscale/Normalize",
            input_type=DataType.IMAGE,
            output_type=DataType.IMAGE,
            steps=steps_cfg,
        )

        if st.button("Traiter l'image", type="primary"):
            with st.spinner('Traitement en cours...'):
                # Sauvegarder l'image temporairement
                temp_dir = os.path.join(tempfile.gettempdir(), 'uploaded_images')
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, file.name)
                original.save(temp_path)
                
                # Appeler le pipeline de traitement
                result = process_image_with_pipeline(temp_path)
                
                if result['success']:
                    st.success('Traitement terminé avec succès !')
                    
                    # Afficher les résultats dans une popup
                    with st.expander("Résultats du traitement", expanded=True):
                        st.json(result['data'])
                        
                        # Si des images de sortie sont disponibles, les afficher
                        if 'output_dir' in result['data'] and os.path.exists(result['data']['output_dir']):
                            output_dir = result['data']['output_dir']
                            st.subheader("Images générées")
                            for img_file in Path(output_dir).glob('*.png'):
                                st.image(str(img_file), caption=img_file.name, use_column_width=True)
                        
                        # Afficher les métadonnées
                        if 'metrics' in result['data']:
                            st.subheader("Métriques")
                            st.json(result['data']['metrics'])
                else:
                    st.error(f"Erreur lors du traitement: {result.get('error', 'Erreur inconnue')}")

            # Afficher le schéma JSON si nécessaire
            if st.checkbox("Afficher le schéma JSON"):
                display_json_schema()

elif mode == "Texte":
    txt = st.text_area("Entrer du texte", height=180)
    if st.button("Traiter le texte", type="primary"):
        text_processed = (txt or "").strip()
        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "length": len(text_processed.encode("utf-8")),
            "words": len(text_processed.split()),
        }
        # Exemple de "normalisation": uppercase
        normalized_text = text_processed.upper()

        # JSON + compression
        result_json = {
            "input_preview": text_processed[:120],
            "normalized_preview": normalized_text[:120],
            "metadata": meta,
        }
        json_bytes = json.dumps(result_json, ensure_ascii=False).encode("utf-8")
        compressed_json_b64 = compressor.compress_to_base64(json_bytes)

        st.subheader("Résultats du pipeline (Texte)")
        st.markdown("**Texte normalisé (aperçu)**")
        st.code(normalized_text[:500] + ("…" if len(normalized_text) > 500 else ""))

        st.markdown("**JSON du résultat**")
        st.code(json.dumps(result_json, indent=2, ensure_ascii=False), language="json")
        st.caption(f"Taille JSON: {len(json_bytes)} octets · Taille compressée (base64): {len(compressed_json_b64)} caractères")

        st.markdown("**JSON compressé (zlib + base64)**")
        st.text_area("Base64", compressed_json_b64, height=160)

        display_json_schema()
