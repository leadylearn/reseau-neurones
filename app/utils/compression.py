"""
Module de compression/décompression des données.
"""
import zlib
import json
import base64
from typing import Union, Dict, Any, Optional
from pathlib import Path
from ..utils.exceptions import CompressionError

class DataCompressor:
    """Classe utilitaire pour la compression des données."""
    
    @staticmethod
    def compress_data(data: Union[dict, str, bytes], compression_level: int = 6) -> bytes:
        """
        Compresse les données avec zlib.
        
        Args:
            data: Données à compresser (dict, str ou bytes)
            compression_level: Niveau de compression (0-9)
            
        Returns:
            Données compressées en bytes
        """
        try:
            if isinstance(data, dict):
                data = json.dumps(data).encode('utf-8')
            elif isinstance(data, str):
                data = data.encode('utf-8')
                
            return zlib.compress(data, level=compression_level)
        except Exception as e:
            raise CompressionError(f"Erreur lors de la compression: {str(e)}")
    
    @staticmethod
    def decompress_data(compressed_data: bytes) -> Union[dict, str, bytes]:
        """
        Décompresse les données.
        
        Args:
            compressed_data: Données compressées
            
        Returns:
            Données décompressées (essaye de convertir en dict si possible)
        """
        try:
            decompressed = zlib.decompress(compressed_data)
            
            # Essayer de décoder en JSON
            try:
                return json.loads(decompressed.decode('utf-8'))
            except:
                # Retourner les bytes bruts si pas du JSON
                try:
                    return decompressed.decode('utf-8')
                except:
                    return decompressed
                    
        except Exception as e:
            raise CompressionError(f"Erreur lors de la décompression: {str(e)}")
    
    @staticmethod
    def compress_to_base64(data: Union[dict, str, bytes]) -> str:
        """
        Compresse les données et les encode en base64.
        
        Args:
            data: Données à compresser
            
        Returns:
            Données compressées encodées en base64
        """
        try:
            compressed = DataCompressor.compress_data(data)
            return base64.b64encode(compressed).decode('utf-8')
        except Exception as e:
            raise CompressionError(f"Erreur lors de l'encodage base64: {str(e)}")
    
    @staticmethod
    def decompress_from_base64(encoded_data: str) -> Union[dict, str, bytes]:
        """
        Décode et décompresse les données depuis base64.
        
        Args:
            encoded_data: Données encodées en base64
            
        Returns:
            Données décompressées
        """
        try:
            compressed = base64.b64decode(encoded_data)
            return DataCompressor.decompress_data(compressed)
        except Exception as e:
            raise CompressionError(f"Erreur lors du décodage base64: {str(e)}")
    
    @staticmethod
    def compress_file(input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Compresse un fichier.
        
        Args:
            input_path: Chemin du fichier source
            output_path: Chemin de sortie (optionnel)
            
        Returns:
            Chemin du fichier compressé
        """
        try:
            input_path = Path(input_path)
            if not input_path.exists():
                raise FileNotFoundError(f"Le fichier source n'existe pas: {input_path}")
                
            if output_path is None:
                output_path = input_path.with_suffix(f"{input_path.suffix}.zlib")
            else:
                output_path = Path(output_path)
                
            with open(input_path, 'rb') as f_in:
                data = f_in.read()
                
            compressed = DataCompressor.compress_data(data)
            
            with open(output_path, 'wb') as f_out:
                f_out.write(compressed)
                
            return output_path
            
        except Exception as e:
            raise CompressionError(f"Erreur lors de la compression du fichier: {str(e)}")
    
    @staticmethod
    def decompress_file(input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Décompresse un fichier.
        
        Args:
            input_path: Chemin du fichier compressé
            output_path: Chemin de sortie (optionnel)
            
        Returns:
            Chemin du fichier décompressé
        """
        try:
            input_path = Path(input_path)
            if not input_path.exists():
                raise FileNotFoundError(f"Le fichier compressé n'existe pas: {input_path}")
                
            if output_path is None:
                if input_path.suffix == '.zlib':
                    output_path = input_path.with_suffix('')
                else:
                    output_path = input_path.with_suffix(f"{input_path.suffix}.decompressed")
            else:
                output_path = Path(output_path)
            
            with open(input_path, 'rb') as f_in:
                compressed_data = f_in.read()
                
            decompressed = DataCompressor.decompress_data(compressed_data)
            
            if isinstance(decompressed, (dict, list)):
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    json.dump(decompressed, f_out, ensure_ascii=False, indent=2)
            elif isinstance(decompressed, str):
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(decompressed)
            else:  # bytes
                with open(output_path, 'wb') as f_out:
                    f_out.write(decompressed)
                    
            return output_path
            
        except Exception as e:
            raise CompressionError(f"Erreur lors de la décompression du fichier: {str(e)}")
