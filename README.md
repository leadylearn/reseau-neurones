# Simulation de Réseaux de Neurones 2D

API pour la simulation de réseaux de neurones en 2D, construite avec FastAPI.

## Structure du Projet

```
project_root/
├── app/
│   ├── __init__.py
│   ├── main.py          # Point d'entrée FastAPI
│   ├── schemas/         # Schémas Pydantic
│   ├── services/        # Logique métier
│   ├── models/          # Modèles de données
│   ├── utils/           # Utilitaires
│   └── api/             # Routeurs API
└── tests/               # Tests unitaires
```

## Installation

1. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: .\venv\Scripts\activate
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Lancement

```bash
uvicorn app.main:app --reload
```

L'API sera disponible à l'adresse : http://127.0.0.1:8000

## Documentation

- Documentation interactive : http://127.0.0.1:8000/docs
- Documentation alternative : http://127.0.0.1:8000/redoc

## Endpoints Principaux

- `GET /` - Page d'accueil
- `POST /train/` - Entraîner un modèle
- `GET /models/{model_id}` - Récupérer les informations d'un modèle
