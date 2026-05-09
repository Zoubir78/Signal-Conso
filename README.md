# Signal-Conso

Plateforme intelligente de traitement automatique des signalements et de priorisation des demandes clients.

## Présentation

**Signal-Conso** est une plateforme data & IA conçue pour automatiser le traitement des signalements clients.  
Elle permet de collecter, nettoyer, classer et prioriser les demandes, puis d’exposer les prédictions via une API et un dashboard de suivi.

## Objectifs

Ce projet a pour objectifs de :

- collecter et préparer les signalements clients ;
- classifier automatiquement les demandes par catégorie ;
- estimer leur niveau de priorité ;
- exposer les prédictions via une API ;
- suivre le pipeline de traitement et les performances des modèles ;
- automatiser les workflows avec des outils d’orchestration ;
- structurer une base solide pour l’industrialisation du projet.

## Fonctionnalités

- ingestion et préparation des données ;
- nettoyage, normalisation et enrichissement des signalements ;
- stockage analytique dans BigQuery ;
- transformation et modélisation avec dbt ;
- entraînement et évaluation des modèles ;
- exposition des prédictions via FastAPI ;
- visualisation et suivi via Streamlit ;
- orchestration des workflows via Prefect Cloud.

## Stack technique

| Domaine | Outils |
|---|---|
| Versioning & CI/CD | GitHub, GitHub Actions |
| Qualité de code | Ruff, SQLFluff, pre-commit |
| Conteneurisation | Docker, Docker Compose |
| Infrastructure | GCS |
| Stockage & transformation | BigQuery, dbt-bigquery |
| Orchestration | Prefect Cloud |
| API de prédiction | FastAPI |
| Visualisation | Streamlit |

## Architecture

- **Ingestion** : récupération des données et contrôles qualité
- **Transformation** : nettoyage, normalisation et préparation des jeux de données
- **Stockage analytique** : entrepôt de données dans BigQuery
- **Modélisation** : entraînement et évaluation du modèle de classification / priorisation
- **API** : exposition des prédictions via FastAPI
- **Orchestration** : automatisation des workflows avec Prefect Cloud
- **Dashboard** : visualisation, suivi des métriques et monitoring

## Structure des données

Le dépôt ne versionne pas les fichiers volumineux.

- `data/raw/` : données brutes locales ou temporaires
- `data/processed/` : données nettoyées et préparées
- `data/models/` : modèles entraînés
- GCS : stockage des jeux de données, modèles et prédictions

## Prérequis

- Python 3.11+
- Docker
- Compte GCP configuré
- `gcloud` installé
- Accès à BigQuery et GCS

## Installation

### 1. Créer l’environnement Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configurer dbt

```bash
# Authentification GCP
gcloud auth application-default login

# Copier le fichier de configuration dbt
cp dbt/profiles.yml.example ~/.dbt/profiles.yml

# Vérifier la connexion
cd dbt && dbt debug
```

### 3. Lancer les transformations dbt

```bash
dbt run
dbt test
```

### 4. Lancer l’API

```bash
uvicorn src.app.api.main:app --reload
```

### 5. Lancer le dashboard Streamlit

```bash
streamlit run src/app/dashboard/dashboard.py
```

### 6. Lancer l’ensemble avec Docker

```bash
docker compose up --build
```

## Pipeline principal

1. extraction des signalements ;
2. nettoyage et transformation ;
3. chargement dans BigQuery ;
4. modélisation et enrichissement via dbt ;
5. entraînement et évaluation du modèle ;
6. exposition des prédictions via l’API ;
7. orchestration des flows avec Prefect ;
8. visualisation et suivi dans Streamlit.

## API

### Endpoints principaux

- `GET /health` : vérification de l’état de l’API
- `POST /predictions` : génération d’une prédiction
- `GET /predictions/{id}` : récupération d’une prédiction par identifiant
- `POST /flows` : déclenchement d’un workflow
