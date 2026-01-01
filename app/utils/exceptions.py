"""
Exceptions personnalisées pour le système de traitement
"""

class ProcessingError(Exception):
    """Classe de base pour les erreurs de traitement"""
    def __init__(self, message: str, error_code: str = "PROCESSING_ERROR", status_code: int = 400):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)

class ValidationError(ProcessingError):
    """Erreur de validation des données d'entrée"""
    def __init__(self, message: str = "Erreur de validation des données"):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422
        )

class CompressionError(ProcessingError):
    """Erreur lors de la compression/décompression des données"""
    def __init__(self, message: str = "Erreur de compression/décompression"):
        super().__init__(
            message=message,
            error_code="COMPRESSION_ERROR",
            status_code=500
        )

class SerializationError(ProcessingError):
    """Erreur lors de la sérialisation/désérialisation"""
    def __init__(self, message: str = "Erreur de sérialisation"):
        super().__init__(
            message=message,
            error_code="SERIALIZATION_ERROR",
            status_code=500
        )
