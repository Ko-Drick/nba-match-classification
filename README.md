# NBA Match Classification

Projet universitaire dans le cadre du cours de **Fouille de données**.  
Objectif : prédire le résultat (victoire à domicile ou non) de matchs NBA à partir de statistiques historiques, en comparant plusieurs algorithmes de classification.

## Dataset

Source : [Kaggle – NBA Games Dataset](https://www.kaggle.com/datasets/nathanlauga/nba-games)  
Fichier utilisé : `csv/game.csv`  
Chaque ligne représente un match NBA avec les statistiques des deux équipes (points, passes décisives, pertes de balle, +/−, etc.).

## Features

Les features sont construites dynamiquement à partir des **5 derniers matchs** de chaque équipe avant la rencontre, afin d'éviter toute fuite de données :

| Feature | Description |
|---|---|
| `home/away_win_rate_last_5` | Taux de victoire sur les 5 derniers matchs |
| `home/away_pts_avg_last_5` | Moyenne de points sur les 5 derniers matchs |
| `home/away_ast_tov_ratio_last_5` | Ratio passes décisives / pertes de balle |
| `home/away_plus_minus_avg_last_5` | Différentiel moyen de points |
| `diff_*` | Différences domicile − extérieur pour chaque stat |
| `home/away_elo` | Score ELO de chaque équipe |
| `diff_elo` | Différence de score ELO |

**Total : 15 features**, normalisées par Z-score.

## Modèles comparés

- **K-NN** — avec recherche du meilleur K par cross-validation (K=228)
- **SVM** (noyau linéaire) — avec recherche du meilleur C (C=0.1)
- **Random Forest** — avec recherche des meilleurs hyperparamètres (n=300, depth=10)
- **Régression Logistique** — avec recherche du meilleur C (C=0.1)

La sélection des hyperparamètres est faite par **cross-validation 5-fold stratifiée** sur le jeu d'entraînement.

## Résultats

| Modèle | Train (CV) | Test | Écart |
|---|---|---|---|
| Logistic Regression (C=0.1) | ~68.6% | ~65.9% | −2.7% |
| Random Forest (n=300) | ~68.5% | ~65.7% | −2.8% |
| SVM (C=0.1) | ~68.5% | ~64.7% | −3.8% |
| K-NN (K=228) | ~68.2% | ~65.1% | −3.1% |
| **Baseline** (toujours domicile) | — | ~59.9% | — |

Tous les modèles améliorent la baseline d'environ **+5–6 points de pourcentage**.

## Structure du projet

```
├── script.py        # Code principal (chargement, features, modèles, évaluation)
├── csv/
│   └── game.csv     # Dataset Kaggle (non inclus dans le dépôt)
└── README.md
```

## Utilisation

```bash
pip install scikit-learn numpy matplotlib
python script.py
```

> Le fichier `csv/game.csv` doit être téléchargé depuis Kaggle et placé dans le dossier `csv/`.
