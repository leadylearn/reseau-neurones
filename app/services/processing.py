import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic, Type, Union
from datetime import datetime
import uuid
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app.schemas.processing import (
    ProcessingStatus,
    ProcessingStep,
    ProcessingPipelineConfig,
    ProcessingRequest,
    ProcessingResponse,
    BatchProcessingRequest,
    BatchProcessingResponse,
    DataType
)
from app.utils.compression import DataCompressor
from app.utils.exceptions import CompressionError

# Type variable pour les données d'entrée/sortie
T = TypeVar('T')
R = TypeVar('R')

# Type pour les fonctions de traitement
ProcessingFunction = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]

class ProcessingError(Exception):
    """Exception personnalisée pour les erreurs de traitement"""
    def __init__(self, message: str, step_name: str = None, original_error: Exception = None):
        self.message = message
        self.step_name = step_name
        self.original_error = original_error
        super().__init__(self.message)

class ProcessingStepExecutor(Generic[T, R]):
    """Exécute une étape de traitement individuelle"""
    
    def __init__(self, step: ProcessingStep, processing_func: ProcessingFunction):
        self.step = step
        self.processing_func = processing_func
        self.logger = logging.getLogger(__name__)
    
    def execute(self, data: T) -> R:
        """Exécute l'étape de traitement"""
        try:
            self.logger.info(f"Exécution de l'étape: {self.step.name}")
            result = self.processing_func(data, self.step.parameters)
            return result
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution de l'étape {self.step.name}: {str(e)}")
            raise ProcessingError(
                message=f"Échec de l'étape {self.step.name}: {str(e)}",
                step_name=self.step.name,
                original_error=e
            )

class ProcessingPipeline(Generic[T, R]):
    """Pipeline de traitement de données générique"""
    
    def __init__(self, config: ProcessingPipelineConfig):
        self.config = config
        self.steps: Dict[str, ProcessingStepExecutor] = {}
        self.logger = logging.getLogger(__name__)
        self._validate_config()
    
    def _validate_config(self):
        """Valide la configuration du pipeline"""
        if not self.config.steps:
            raise ValueError("Le pipeline doit contenir au moins une étape")
    
    def add_step(self, step_name: str, processing_func: ProcessingFunction) -> None:
        """Ajoute une étape au pipeline"""
        step = next((s for s in self.config.steps if s.name == step_name), None)
        if not step:
            raise ValueError(f"Aucune étape nommée '{step_name}' trouvée dans la configuration")
        
        self.steps[step_name] = ProcessingStepExecutor(step, processing_func)
    
    def process(self, input_data: T) -> R:
        """Exécute le pipeline de traitement"""
        if not self.steps:
            raise ValueError("Aucune étape n'a été configurée pour ce pipeline")
        
        current_data = input_data
        
        # Exécute chaque étape dans l'ordre défini
        for step in sorted(self.config.steps, key=lambda x: x.order):
            if step.name not in self.steps:
                if step.required:
                    raise ValueError(f"Étape requise manquante: {step.name}")
                continue
                
            executor = self.steps[step.name]
            try:
                current_data = executor.execute(current_data)
            except ProcessingError as e:
                self.logger.error(f"Échec du pipeline à l'étape {step.name}: {str(e)}")
                raise
        
        return current_data

class ProcessingService:
    """Service de gestion des traitements"""
    
    def __init__(self):
        self.pipelines: Dict[str, ProcessingPipeline] = {}
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.compressor = DataCompressor()
        
    def _compress_large_data(self, data: Any, threshold: int = 1024) -> Dict[str, Any]:
        """
        Compresse les données si elles dépassent la taille seuil.
        
        Args:
            data: Données à potentiellement compresser
            threshold: Taille seuil en octets au-delà de laquelle on compresse
            
        Returns:
            Dictionnaire contenant les données (compressées ou non) et des métadonnées
        """
        try:
            # Conversion en JSON pour estimer la taille
            json_data = json.dumps(data).encode('utf-8')
            
            if len(json_data) > threshold:
                compressed = self.compressor.compress_to_base64(json_data)
                return {
                    'compressed': True,
                    'data': compressed,
                    'original_size': len(json_data),
                    'compressed_size': len(compressed),
                    'compression_ratio': len(compressed) / len(json_data) if json_data else 0
                }
            
            return {'compressed': False, 'data': data}
            
        except Exception as e:
            self.logger.warning(f"Échec de la compression des données: {str(e)}")
            return {'compressed': False, 'data': data, 'error': str(e)}
    
    def _decompress_data_if_needed(self, data: Dict[str, Any]) -> Any:
        """
        Décompresse les données si elles ont été compressées.
        
        Args:
            data: Données potentiellement compressées
            
        Returns:
            Données décompressées
        """
        try:
            if not isinstance(data, dict) or not data.get('compressed', False):
                return data
                
            if 'data' not in data:
                raise ValueError("Données compressées invalides: clé 'data' manquante")
                
            if data['compressed']:
                return self.compressor.decompress_from_base64(data['data'])
                
            return data['data']
            
        except Exception as e:
            self.logger.error(f"Échec de la décompression des données: {str(e)}")
            raise CompressionError(f"Impossible de décompresser les données: {str(e)}")
    
    def register_pipeline(self, pipeline_config: ProcessingPipelineConfig) -> None:
        """Enregistre un nouveau pipeline de traitement"""
        self.pipelines[pipeline_config.pipeline_id] = ProcessingPipeline(pipeline_config)
        self.logger.info(f"Pipeline enregistré: {pipeline_config.pipeline_id}")
    
    def get_pipeline(self, pipeline_id: str) -> ProcessingPipeline:
        """Récupère un pipeline par son ID"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline non trouvé: {pipeline_id}")
        return self.pipelines[pipeline_id]
    
    async def process(self, request: ProcessingRequest) -> ProcessingResponse:
        """
        Traite une requête de traitement avec gestion de la compression des données volumineuses.
        
        Args:
            request: Requête de traitement
            
        Returns:
            Réponse de traitement avec les résultats
        """
        request_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            # Validation de la requête
            if not request.pipeline_id:
                raise ProcessingError("L'ID du pipeline est requis")
                
            if not request.input_data:
                raise ProcessingError("Les données d'entrée sont requises")
            
            # Traitement des paramètres de compression
            compression_params = request.parameters.get('compression', {})
            compress_threshold = compression_params.get('threshold', 1024)  # 1KB par défaut
            
            # Compression des données d'entrée si nécessaire
            input_data = request.input_data
            if isinstance(input_data, dict) and not input_data.get('compressed', False):
                input_data = self._compress_large_data(input_data, compress_threshold)
            
            # Récupération du pipeline
            pipeline = self.get_pipeline(request.pipeline_id)
            
            # Préparation des paramètres avec les données d'entrée potentiellement compressées
            processing_params = request.parameters.copy()
            processing_params['input_data'] = input_data
            
            # Exécution du traitement
            result = await self._run_in_executor(
                pipeline.process,
                **processing_params
            )
            
            # Décompression du résultat si nécessaire
            if isinstance(result, dict) and result.get('compressed', False):
                try:
                    result['data'] = self._decompress_data_if_needed(result)
                except Exception as e:
                    self.logger.error(f"Échec de la décompression du résultat: {str(e)}")
            
            # Calcul du temps de traitement
            processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Création de la réponse
            return ProcessingResponse(
                request_id=request_id,
                status=ProcessingStatus.COMPLETED,
                pipeline_id=request.pipeline_id,
                result=result,
                metadata={
                    "processing_time_ms": processing_time_ms,
                    "steps_executed": len(pipeline.steps),
                    "compression": {
                        "input_compressed": input_data.get('compressed', False),
                        "output_compressed": isinstance(result, dict) and result.get('compressed', False),
                        "processing_time_ms": processing_time_ms
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement de la requête {request_id}: {str(e)}", exc_info=True)
            return ProcessingResponse(
                request_id=request_id,
                status=ProcessingStatus.FAILED,
                pipeline_id=request.pipeline_id if 'request' in locals() else None,
                error=str(e),
                error_code=getattr(e, 'error_code', 'PROCESSING_ERROR'),
                metadata={
                    "error_type": e.__class__.__name__,
                    "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                    "error_details": {
                        "message": str(e),
                        "type": e.__class__.__name__,
                        "code": getattr(e, 'error_code', 'UNKNOWN')
                    }
                }
            )
    
    async def process_batch(self, batch_request: BatchProcessingRequest) -> BatchProcessingResponse:
        """Traite un lot de requêtes"""
        batch_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        results = []
        completed = 0
        failed = 0
        
        # Exécute les requêtes en parallèle
        futures = []
        for req in batch_request.requests:
            future = self.executor.submit(
                self.process,
                req
            )
            futures.append(future)
        
        # Attend que toutes les requêtes soient terminées
        for future in as_completed(futures):
            try:
                result = await future.result()
                if result.status == ProcessingStatus.COMPLETED:
                    completed += 1
                else:
                    failed += 1
                results.append(result)
            except Exception as e:
                failed += 1
                self.logger.error(f"Erreur lors du traitement d'une requête du lot {batch_id}: {str(e)}")
        
        # Détermine le statut global
        if failed == 0:
            status = ProcessingStatus.COMPLETED
        elif completed == 0:
            status = ProcessingStatus.FAILED
        else:
            status = ProcessingStatus.COMPLETED  # Partiellement réussi
        
        return BatchProcessingResponse(
            batch_id=batch_id,
            status=status,
            total_requests=len(batch_request.requests),
            completed_requests=completed,
            failed_requests=failed,
            results=results,
            metadata={
                "processing_time_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "batch_size": len(batch_request.requests)
            }
        )
    
    async def _run_in_executor(self, func, *args):
        """Exécute une fonction synchrone dans un thread séparé"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, func, *args)
    
    def __del__(self):
        """Nettoie les ressources lors de la destruction du service"""
        self.executor.shutdown(wait=True)
