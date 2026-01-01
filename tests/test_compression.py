"""
Tests pour le module de compression/décompression
"""
import os
import json
import pytest
import numpy as np
from pathlib import Path
from datetime import datetime

from app.utils.compression import DataCompressor
from app.utils.exceptions import CompressionError
from datetime import datetime, timezone

def test_compress_decompress_dict():
    """Test la compression et décompression d'un dictionnaire"""
    compressor = DataCompressor()
    test_data = {
        "name": "Test",
        "value": 42,
        "nested": {"key": "value"},
        "list": [1, 2, 3],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Compression
    compressed = compressor.compress_data(test_data)
    assert isinstance(compressed, bytes)
    assert len(compressed) > 0
    
    # Décompression
    decompressed = compressor.decompress_data(compressed)
    assert decompressed == test_data

def test_compress_decompress_string():
    """Test la compression et décompression d'une chaîne de caractères"""
    compressor = DataCompressor()
    test_string = "Ceci est un test de compression de chaîne de caractères" * 10
    
    # Compression
    compressed = compressor.compress_data(test_string)
    assert isinstance(compressed, bytes)
    
    # Décompression
    decompressed = compressor.decompress_data(compressed)
    assert decompressed == test_string

def test_compress_decompress_base64():
    """Test la compression et décompression avec encodage base64"""
    compressor = DataCompressor()
    test_data = {"key": "value" * 100}  # Données assez grandes pour être compressées
    
    # Compression en base64
    encoded = compressor.compress_to_base64(test_data)
    assert isinstance(encoded, str)
    
    # Décompression depuis base64
    decoded = compressor.decompress_from_base64(encoded)
    assert decoded == test_data

def test_compress_decompress_file(tmp_path):
    """Test la compression et décompression d'un fichier"""
    compressor = DataCompressor()
    
    # Création d'un fichier de test
    test_file = tmp_path / "test_file.txt"
    test_content = "Contenu de test" * 1000
    test_file.write_text(test_content, encoding='utf-8')
    
    # Compression du fichier
    compressed_file = compressor.compress_file(test_file)
    assert compressed_file.exists()
    assert compressed_file.suffix == ".zlib"
    
    # Décompression du fichier
    decompressed_file = compressor.decompress_file(compressed_file)
    assert decompressed_file.exists()
    assert decompressed_file.read_text(encoding='utf-8') == test_content

def test_compress_empty_data():
    """Test la compression de données vides"""
    compressor = DataCompressor()
    
    # Test avec un dictionnaire vide
    compressed = compressor.compress_data({})
    assert compressor.decompress_data(compressed) == {}
    
    # Test avec une chaîne vide
    compressed = compressor.compress_data("")
    assert compressor.decompress_data(compressed) == ""

def test_compress_large_data():
    """Test la compression de données volumineuses"""
    compressor = DataCompressor()
    
    # Création de données volumineuses
    large_data = {
        f"key_{i}": f"value_{i}" * 100 
        for i in range(1000)
    }
    
    # Compression
    compressed = compressor.compress_data(large_data)
    decompressed = compressor.decompress_data(compressed)
    
    # Vérification que les données décompressées sont identiques aux originales
    assert decompressed == large_data
    
    # Vérification que la compression réduit effectivement la taille
    original_size = len(json.dumps(large_data).encode('utf-8'))
    compressed_size = len(compressed)
    assert compressed_size < original_size  # La compression doit réduire la taille

def test_compress_with_custom_level():
    """Test la compression avec différents niveaux de compression"""
    compressor = DataCompressor()
    test_data = {"key": "value" * 1000}
    
    # Compression avec différents niveaux
    compressed_fast = compressor.compress_data(test_data, compression_level=1)  # Rapide, compression minimale
    compressed_best = compressor.compress_data(test_data, compression_level=9)  # Lent, meilleure compression
    
    # Vérification que le niveau de compression affecte la taille du résultat
    assert len(compressed_best) <= len(compressed_fast)
    
    # Vérification que la décompression fonctionne dans les deux cas
    assert compressor.decompress_data(compressed_fast) == test_data
    assert compressor.decompress_data(compressed_best) == test_data

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_compression.py"])
