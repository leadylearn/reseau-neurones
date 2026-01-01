from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, validator, root_validator

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DataType(str, Enum):
    IMAGE = "image"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"

class ProcessingStep(BaseModel):
    """
    Représente une étape de traitement dans le pipeline
    """
    name: str = Field(..., description="Nom de l'étape de traitement")
    description: str = Field(default="", description="Description de l'étape")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Paramètres de l'étape")
    order: int = Field(..., description="Ordre d'exécution de l'étape")
    required: bool = Field(default=True, description="Si l'étape est obligatoire")

class ProcessingPipelineConfig(BaseModel):
    """
    Configuration complète d'un pipeline de traitement
    """
    pipeline_id: str = Field(..., description="Identifiant unique du pipeline")
    name: str = Field(..., description="Nom du pipeline")
    description: str = Field(default="", description="Description du pipeline")
    version: str = Field(default="1.0.0", description="Version du pipeline")
    input_type: DataType = Field(..., description="Type de données en entrée")
    output_type: DataType = Field(..., description="Type de données en sortie")
    steps: List[ProcessingStep] = Field(..., description="Liste des étapes de traitement")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Date de création")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Dernière mise à jour")

    @validator('steps')
    def validate_steps_order(cls, steps):
        """Valide que les étapes ont des ordres uniques et séquentiels"""
        orders = [step.order for step in steps]
        if len(orders) != len(set(orders)):
            raise ValueError("Les étapes doivent avoir des ordres uniques")
        if sorted(orders) != list(range(1, len(orders) + 1)):
            raise ValueError("Les étapes doivent avoir des ordres séquentiels commençant à 1")
        return steps

class ProcessingRequest(BaseModel):
    """
    Requête de traitement de données
    """
    pipeline_id: str = Field(..., description="ID du pipeline à utiliser")
    input_data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(..., description="Données d'entrée")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Paramètres additionnels"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "pipeline_id": "image_processing",
                "input_data": {
                    "image": "base64_encoded_image_data",
                    "metadata": {"source": "test"}
                },
                "parameters": {
                    "compression": {
                        "enabled": True,
                        "threshold": 1024,
                        "algorithm": "zlib"
                    },
                    "quality": 90
                }
            }
        }

class CompressionInfo(BaseModel):
    """
    Informations sur la compression des données
    """
    compressed: bool = Field(False, description="Si les données sont compressées")
    algorithm: Optional[str] = Field(None, description="Algorithme de compression utilisé")
    original_size: Optional[int] = Field(None, description="Taille originale des données en octets")
    compressed_size: Optional[int] = Field(None, description="Taille compressée des données en octets")
    compression_ratio: Optional[float] = Field(None, description="Taux de compression (0-1)")
    
    @root_validator(pre=True)
    def calculate_ratio(cls, values):
        if values.get('original_size') and values.get('compressed_size'):
            values['compression_ratio'] = values['compressed_size'] / values['original_size']
        return values

class ProcessingResponse(BaseModel):
    """
    Réponse du traitement
    """
    request_id: str = Field(..., description="ID de la requête")
    status: ProcessingStatus = Field(..., description="Statut du traitement")
    pipeline_id: str = Field(..., description="ID du pipeline utilisé")
    result: Optional[Dict[str, Any]] = Field(None, description="Résultat du traitement")
    error: Optional[str] = Field(None, description="Message d'erreur en cas d'échec")
    error_code: Optional[str] = Field(None, description="Code d'erreur standardisé")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Métadonnées additionnelles"
    )
    compression: Optional[CompressionInfo] = Field(
        None,
        description="Informations sur la compression des données"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Date de création")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Dernière mise à jour")

class BatchProcessingRequest(BaseModel):
    """
    Requête de traitement par lots
    """
    requests: List[ProcessingRequest] = Field(..., description="Liste des requêtes de traitement")
    priority: int = Field(default=0, ge=0, le=10, description="Priorité du traitement (0-10)")

class BatchProcessingResponse(BaseModel):
    """
    Réponse du traitement par lots
    """
    batch_id: str = Field(..., description="ID du lot de traitement")
    status: ProcessingStatus = Field(..., description="Statut global du traitement")
    total_requests: int = Field(..., description="Nombre total de requêtes")
    completed_requests: int = Field(0, description="Nombre de requêtes terminées")
    failed_requests: int = Field(0, description="Nombre de requêtes échouées")
    results: List[ProcessingResponse] = Field(default_factory=list, description="Résultats individuels")
