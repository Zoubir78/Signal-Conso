# Architecture du projet Signal-Conso

## Vue d’ensemble

Signal-Conso est une plateforme data & IA conçue pour automatiser le traitement des signalements clients, depuis leur extraction jusqu’à la visualisation des résultats.
L’architecture s’appuie sur une chaîne de traitement orientée **data engineering**, **machine learning** et **MLOps**.

## Pipeline principal

Le pipeline du projet suit les étapes suivantes :

1. **Extraction des signalements**
   Collecte des signalements depuis les sources amont.

2. **Nettoyage et transformation**
   Normalisation, déduplication, structuration et enrichissement initial des données.

3. **Chargement dans BigQuery**
   Stockage des données préparées dans l’entrepôt analytique.

4. **Modélisation et enrichissement via dbt**
   Construction des couches de données, transformation SQL et tests de qualité.

5. **Entraînement et évaluation du modèle**
   Création des features, entraînement du modèle et suivi des métriques.

6. **Exposition des prédictions via l’API**
   Mise à disposition des prédictions et des scores via FastAPI.

7. **Orchestration des flows avec Prefect**
   Automatisation, planification et supervision des traitements.

8. **Visualisation et suivi dans Streamlit**
   Dashboard de suivi pour les indicateurs, les prédictions et l’état du pipeline.

## Architecture fonctionnelle

### 1. Ingestion
Cette couche récupère les signalements et prépare les données brutes pour la suite du traitement.

### 2. Préparation des données
Les données sont nettoyées, normalisées et transformées afin de garantir leur qualité et leur exploitabilité.

### 3. Stockage analytique
BigQuery sert de socle central pour le stockage et l’analyse des données structurées.

### 4. Transformation métier
dbt permet de construire des tables intermédiaires et des tables métier réutilisables.

### 5. Machine Learning
Le modèle est entraîné sur les jeux de données préparés afin de produire des prédictions de catégorie ou de priorité.

### 6. API de prédiction
FastAPI expose les résultats du modèle à des consommateurs internes ou externes.

### 7. Orchestration
Prefect orchestre les workflows, supervise l’exécution des tâches et facilite la reprise en cas d’erreur.

### 8. Dashboard
Streamlit fournit une interface de visualisation pour suivre les métriques, les volumes et les prédictions.

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

## Organisation des données

Le dépôt ne versionne pas les fichiers volumineux.

- `data/raw/` : données brutes locales ou temporaires
- `data/processed/` : données nettoyées et préparées
- `data/models/` : modèles entraînés
- **GCS** : stockage des jeux de données, modèles et prédictions

## Arborescence cible

```text
signal-conso/
├── docs/
│   └── architecture.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── dbt/
├── src/
│   ├── app/
│   │   ├── api/
│   │   └── dashboard/
│   └── pipelines/
├── tests/
└── docker-compose.yml
```

## Bonnes pratiques retenues

- séparation claire entre ingestion, transformation, modélisation et exposition ;
- utilisation d’un entrepôt analytique centralisé ;
- contrôle qualité des transformations SQL avec dbt ;
- automatisation des workflows avec Prefect ;
- conteneurisation pour assurer la reproductibilité ;
- visualisation dédiée pour le pilotage opérationnel.

## Objectif de cette architecture

Cette architecture a pour but de fournir une base robuste, lisible et industrialisable pour faire évoluer Signal-Conso vers un produit data fiable, maintenable et scalable.
