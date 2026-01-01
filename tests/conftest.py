"""Configuration des tests et fixtures partagées."""

import pytest
from pathlib import Path
import tempfile
import shutil
from typing import Generator, Dict, Any

@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Crée un répertoire temporaire pour les tests et le supprime après utilisation."""
    temp_dir = tempfile.mkdtemp(prefix="test_compression_")
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    """Crée un fichier image de test et retourne son chemin."""
    # Pour les tests réels, vous pourriez utiliser une vraie image de test
    # Ici, on crée juste un fichier binaire factice
    img_path = tmp_path / "test_image.png"
    with open(img_path, 'wb') as f:
        # En-tête PNG factice
        f.write(b'\x89PNG\r\n\x1a\n')
        # Données d'image factices
        f.write(b'\x00' * 1024)  # 1KB de données
    return img_path

@pytest.fixture
def sample_large_data() -> Dict[str, Any]:
    """Retourne un grand ensemble de données pour les tests de compression."""
    return {
        f"key_{i}": {
            "id": i,
            "name": f"Item {i}",
            "value": i * 1.5,
            "active": i % 2 == 0,
            "tags": [f"tag_{j}" for j in range(5)],
            "metadata": {"created_at": "2023-01-01T00:00:00"}
        }
        for i in range(1000)  # 1000 éléments
    }

@pytest.fixture
def test_compressor():
    """Retourne une instance de DataCompressor pour les tests."""
    from app.utils.compression import DataCompressor
    return DataCompressor()
