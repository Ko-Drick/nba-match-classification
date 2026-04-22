'''Fouille de données – Projet
Prédiction de resultat de matchs NBA
'''
from math import sqrt, log, exp
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def read_data(filename):
  '''
  Lit le fichier "filename" et le transforme en Dictionnaires
  
  :param filename: chemin vers le fichier CSV
  '''
  games = []

  with open(filename, 'r') as f:
    header = f.readline().strip().split(',')
    col_idx = {col: i for i, col in enumerate(header)}
    
    for line in f:
      parts = line.strip().split(',')
      
      if len(parts) < len(header):
        continue
      
      try:
        game = {
          'game_date': parts[col_idx['game_date']],
          'home_team': parts[col_idx['team_abbreviation_home']],
          'away_team': parts[col_idx['team_abbreviation_away']],
          'home_pts': float(parts[col_idx['pts_home']]) if parts[col_idx['pts_home']] else 0,
          'away_pts': float(parts[col_idx['pts_away']]) if parts[col_idx['pts_away']] else 0,
          'home_ast': float(parts[col_idx['ast_home']]) if parts[col_idx['ast_home']] else 0,
          'away_ast': float(parts[col_idx['ast_away']]) if parts[col_idx['ast_away']] else 0,
          'home_tov': float(parts[col_idx['tov_home']]) if parts[col_idx['tov_home']] else 1,
          'away_tov': float(parts[col_idx['tov_away']]) if parts[col_idx['tov_away']] else 1,
          'home_plus_minus': float(parts[col_idx['plus_minus_home']]) if parts[col_idx['plus_minus_home']] else 0,
          'wl_home': parts[col_idx['wl_home']],
          'season_type': parts[col_idx['season_type']] if 'season_type' in col_idx else 'Regular Season'
        }

        if game['home_pts'] > 0 and game['away_pts'] > 0:
          games.append(game)
              
      except (ValueError, IndexError) as e:
        continue

  print(f"    {len(games)} matchs chargés depuis {filename}")
  return games

def split_train_test(games, ratio=0.8):
  '''
  Sépare les données en deux fichiers train et test
  
  :param games: liste des parties
  :param ratio: proportion de partie dans train
  '''
  games_sorted = sorted(games, key=lambda g: g['game_date'])
  split_idx = int(len(games_sorted) * ratio)
  train = games_sorted[:split_idx]
  test = games_sorted[split_idx:]
  print(f"    Train: {len(train)} matchs (jusqu'au {train[-1]['game_date'][:10]})")
  print(f"    Test: {len(test)} matchs (à partir du {test[0]['game_date'][:10]})")
  return train, test

def build_features_dynamic(games):
    from collections import defaultdict
    
    games_sorted = sorted(games, key=lambda g: g['game_date'])

    team_history = defaultdict(lambda: {
        'last_results': [],
        'last_pts': [],
        'last_ast_tov': [],
        'last_plus_minus': [],
        'elo': 1500
    })

    X, Y = [], []

    for g in games_sorted:
        home = g['home_team']
        away = g['away_team']

        h = team_history[home]
        a = team_history[away]

        enough_data = (
            len(h['last_results']) >= 5 and
            len(a['last_results']) >= 5
        )

        if enough_data:
            home_win_rate = sum(h['last_results'][-5:]) / 5
            home_pts_avg = sum(h['last_pts'][-5:]) / 5
            home_ast_tov = sum(h['last_ast_tov'][-5:]) / 5
            home_pm = sum(h['last_plus_minus'][-5:]) / 5

            away_win_rate = sum(a['last_results'][-5:]) / 5
            away_pts_avg = sum(a['last_pts'][-5:]) / 5
            away_ast_tov = sum(a['last_ast_tov'][-5:]) / 5
            away_pm = sum(a['last_plus_minus'][-5:]) / 5

            home_elo = h['elo']
            away_elo = a['elo']
            elo_diff = home_elo - away_elo

            x = [
                home_win_rate, home_pts_avg, home_ast_tov, home_pm,
                away_win_rate, away_pts_avg, away_ast_tov, away_pm,
                home_win_rate - away_win_rate,
                home_pts_avg - away_pts_avg,
                home_ast_tov - away_ast_tov,
                home_pm - away_pm,
                home_elo,
                away_elo,
                elo_diff
            ]

            y = 1 if g['home_pts'] > g['away_pts'] else 0

            X.append(x)
            Y.append(y)

        home_win = 1 if g['home_pts'] > g['away_pts'] else 0
        home_ast_tov_ratio = g['home_ast'] / g['home_tov'] if g['home_tov'] > 0 else 1
        away_ast_tov_ratio = g['away_ast'] / g['away_tov'] if g['away_tov'] > 0 else 1

        new_home_elo, new_away_elo = update_elo(h['elo'], a['elo'], home_win)
        h['elo'] = new_home_elo
        a['elo'] = new_away_elo

        h['last_results'].append(home_win)
        h['last_pts'].append(g['home_pts'])
        h['last_ast_tov'].append(home_ast_tov_ratio)
        h['last_plus_minus'].append(g['home_plus_minus'])

        a['last_results'].append(1 - home_win)
        a['last_pts'].append(g['away_pts'])
        a['last_ast_tov'].append(away_ast_tov_ratio)
        a['last_plus_minus'].append(-g['home_plus_minus'])

    return X, Y

def update_elo(elo_home, elo_away, home_win, K=20, home_adv=100):
    expected_home = 1 / (1 + 10 ** ((elo_away - (elo_home + home_adv)) / 400))
    expected_away = 1 - expected_home
    
    score_home = 1 if home_win else 0
    score_away = 1 - score_home

    new_home = elo_home + K * (score_home - expected_home)
    new_away = elo_away + K * (score_away - expected_away)

    return new_home, new_away

def distance_euclidienne(data1, data2):
  '''
  Calcule la distance euclidienne entre 2 data
  '''
  distance = 0
  for i in range(len(data1)):
    diff = data1[i] - data2[i]
    distance += diff ** 2
  return sqrt(distance)

def normalize(X_train, X_test):
  '''
  Normalisation des features (Z-score)
  '''
  n_features = len(X_train[0])
  n_train = len(X_train)
  means = []
  stds = []
  
  for j in range(n_features):
    values = [X_train[i][j] for i in range(n_train)]
    mean = sum(values) / n_train
    variance = sum((v - mean) ** 2 for v in values) / n_train
    std = sqrt(variance) if variance > 0 else 1
    means.append(mean)
    stds.append(std)
  
  X_train_norm = []
  for x in X_train:
    X_train_norm.append([(x[j] - means[j]) / stds[j] for j in range(n_features)])
  
  X_test_norm = []
  for x in X_test:
    X_test_norm.append([(x[j] - means[j]) / stds[j] for j in range(n_features)])
  
  return X_train_norm, X_test_norm

# ============================================================
# KNN
# ============================================================

def k_nearest_neighbors(x, points, dist_function, k):
  '''
  Renvoie les indices des k points les plus proches de x
  '''
  distances = [(i, dist_function(x, points[i])) for i in range(len(points))]
  distances.sort(key=lambda pair: pair[1])
  return [distances[i][0] for i in range(k)]

def predict_knn(x, train_X, train_Y, dist_function, k):
  '''
  Renvoie 1 si la majorité des voisins sont 1 sinon 0
  '''
  neighbors = k_nearest_neighbors(x, train_X, dist_function, k)
  count = sum(train_Y[i] for i in neighbors)
  return 1 if count > k / 2 else 0

def eval_classifier(test_X, test_Y, classifier):
  '''
  Renvoie le taux d'erreur de notre classifier
  '''
  errors = sum(1 for i in range(len(test_X)) if classifier(test_X[i]) != test_Y[i])
  return errors / len(test_X)

def cross_validation(train_X, train_Y, untrained_classifier):
  '''
  Renvoie la moyenne du taux d'erreur (5-Fold CV)
  '''
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  error_rates = []

  for train_indices, validation_indices in skf.split(train_X, train_Y):
    fold_train_x = [train_X[i] for i in train_indices]
    fold_train_y = [train_Y[i] for i in train_indices]
    
    validation_x = [train_X[i] for i in validation_indices]
    validation_y = [train_Y[i] for i in validation_indices]
    
    classifier = lambda x, ftx=fold_train_x, fty=fold_train_y: untrained_classifier(ftx, fty, x)
    error_rate = eval_classifier(validation_x, validation_y, classifier)
    error_rates.append(error_rate)
  
  return sum(error_rates) / len(error_rates)

def sampled_range(mini, maxi, num):
  if not num:
    return []
  lmini = log(mini)
  lmaxi = log(maxi)
  ldelta = (lmaxi - lmini) / (num - 1)
  out = [x for x in set([int(exp(lmini + i * ldelta)) for i in range(num)])]
  out.sort()
  return out

def find_best_k(train_X, train_Y, untrained_classifier):
  '''
  Renvoie le meilleur k via cross-validation
  '''
  best_k = None
  best_error = float("inf")
  max_k = int(sqrt(len(train_X)))

  k_values = sampled_range(1, max_k, 12)
  print(f"  Test de K de 1 à {max_k}...")

  for k in k_values:
    untrained = lambda tx, ty, x, ck=k: untrained_classifier(tx, ty, ck, x)
    error = cross_validation(train_X, train_Y, untrained)
        
    print(f"  K={k}/{max_k} → erreur CV = {error:.4f}")

    if error < best_error:
      best_error = error
      best_k = k
    
  return (best_k, best_error)

# ============================================================
# SVM
# ============================================================

def SVM_classifier(train_X, train_Y, X, C=1):
  '''
  Fonction de classification SVM
  '''
  svm = SVC(kernel='linear', C=C)
  svm.fit(train_X, train_Y)
  predictions = svm.predict(X)
  return [int(p) for p in predictions]

def cross_validation_svm(train_X, train_Y, C):
  '''
  Cross-validation pour SVM
  '''
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  error_rates = []
  
  train_X_np = np.array(train_X)
  train_Y_np = np.array(train_Y)
  
  for train_idx, val_idx in skf.split(train_X_np, train_Y_np):
    fold_train_X = train_X_np[train_idx]
    fold_train_Y = train_Y_np[train_idx]
    fold_val_X = train_X_np[val_idx]
    fold_val_Y = train_Y_np[val_idx]
    
    svm = SVC(kernel='linear', C=C)
    svm.fit(fold_train_X, fold_train_Y)
    preds = svm.predict(fold_val_X)
    
    err = sum(preds[i] != fold_val_Y[i] for i in range(len(preds))) / len(preds)
    error_rates.append(err)
  
  return sum(error_rates) / len(error_rates)

def find_best_c(train_X, train_Y):
  '''
  Renvoie le meilleur C pour SVM via cross-validation
  '''
  candidate_C = [0.01, 0.1, 1, 10]
  best_C = None
  best_error = float("inf")
  
  print("  Recherche du meilleur C...")
  for C in candidate_C:
    error = cross_validation_svm(train_X, train_Y, C)
    print(f"    C={C}: erreur CV = {error:.4f}")
    
    if error < best_error:
      best_error = error
      best_C = C

  return (best_C, best_error)

# ============================================================
# RANDOM FOREST
# ============================================================

def RF_classifier(train_X, train_Y, test_X, n_estimators=300, max_depth=None):
  '''
  Entraîne un Random Forest et renvoie les prédictions + le modèle
  '''
  model = RandomForestClassifier(
      n_estimators=n_estimators,
      max_depth=max_depth,
      random_state=42,
      min_samples_leaf=5
  )

  model.fit(np.array(train_X), np.array(train_Y))
  predictions = model.predict(np.array(test_X))

  return predictions, model

def cross_validation_rf(train_X, train_Y, n_estimators, max_depth):
  '''
  Cross-validation pour Random Forest
  '''
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  error_rates = []
  
  train_X_np = np.array(train_X)
  train_Y_np = np.array(train_Y)
  
  for train_idx, val_idx in skf.split(train_X_np, train_Y_np):
    fold_train_X = train_X_np[train_idx]
    fold_train_Y = train_Y_np[train_idx]
    fold_val_X = train_X_np[val_idx]
    fold_val_Y = train_Y_np[val_idx]
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        min_samples_leaf=5
    )
    model.fit(fold_train_X, fold_train_Y)
    preds = model.predict(fold_val_X)
    
    err = sum(preds[i] != fold_val_Y[i] for i in range(len(preds))) / len(preds)
    error_rates.append(err)
  
  return sum(error_rates) / len(error_rates)

def find_best_rf_params(train_X, train_Y):
  '''
  Trouve les meilleurs hyperparamètres pour Random Forest via cross-validation
  '''
  n_estimators_list = [100, 200, 300]
  max_depth_list = [10, 20, None]
  
  best_params = None
  best_error = float("inf")
  
  print("  Recherche des meilleurs paramètres RF...")
  for n_est in n_estimators_list:
    for max_d in max_depth_list:
      error = cross_validation_rf(train_X, train_Y, n_est, max_d)
      depth_str = str(max_d) if max_d else "None"
      print(f"    n_estimators={n_est}, max_depth={depth_str}: erreur CV = {error:.4f}")
      
      if error < best_error:
        best_error = error
        best_params = (n_est, max_d, best_error)
  
  return best_params

# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def LR_classifier(train_X, train_Y, test_X, C=1.0):
  '''
  Entraîne une Logistic Regression et renvoie les prédictions + le modèle
  '''
  model = LogisticRegression(C=C, max_iter=1000, random_state=42)
  model.fit(np.array(train_X), np.array(train_Y))
  predictions = model.predict(np.array(test_X))
  
  return predictions, model

def cross_validation_lr(train_X, train_Y, C):
  '''
  Cross-validation pour Logistic Regression
  '''
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  error_rates = []
  
  train_X_np = np.array(train_X)
  train_Y_np = np.array(train_Y)
  
  for train_idx, val_idx in skf.split(train_X_np, train_Y_np):
    fold_train_X = train_X_np[train_idx]
    fold_train_Y = train_Y_np[train_idx]
    fold_val_X = train_X_np[val_idx]
    fold_val_Y = train_Y_np[val_idx]
    
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    model.fit(fold_train_X, fold_train_Y)
    preds = model.predict(fold_val_X)
    
    err = sum(preds[i] != fold_val_Y[i] for i in range(len(preds))) / len(preds)
    error_rates.append(err)
  
  return sum(error_rates) / len(error_rates)

def find_best_lr_C(train_X, train_Y):
  '''
  Trouve le meilleur C pour Logistic Regression via cross-validation
  '''
  candidate_C = [0.01, 0.1, 1, 10, 100]
  best_C = None
  best_error = float("inf")
  
  print("  Recherche du meilleur C...")
  for C in candidate_C:
    error = cross_validation_lr(train_X, train_Y, C)
    print(f"    C={C}: erreur CV = {error:.4f}")
    
    if error < best_error:
      best_error = error
      best_C = C
  
  return (best_C, best_error)

FEATURES_NAME = [
    "home_win_rate_last_5",
    "home_pts_avg_last_5",
    "home_ast_tov_ratio_last_5",
    "home_plus_minus_avg_last_5",
    "away_win_rate_last_5",
    "away_pts_avg_last_5",
    "away_ast_tov_ratio_last_5",
    "away_plus_minus_avg_last_5",
    "diff_win_rate_last_5",
    "diff_pts_avg_last_5",
    "diff_ast_tov_last_5",
    "diff_plus_minus_last_5",
    "home_elo",
    "away_elo",
    "diff_elo"
]

import matplotlib.pyplot as plt

def plot_all_models_comparison(results_test, results_train, baseline):
    '''
    Affiche un graphique comparatif Train (CV) vs Test pour tous les algorithmes
    
    :param results_test: dict {nom_modèle: accuracy_test}
    :param results_train: dict {nom_modèle: accuracy_train_cv}
    :param baseline: valeur de la baseline
    '''
    # Filtrer pour n'avoir que les modèles (pas la baseline)
    models_test = {k: v for k, v in results_test.items() if k != 'Baseline (domicile)'}
    models_train = {k: v for k, v in results_train.items() if k != 'Baseline (domicile)'}
    
    # Trier par accuracy test décroissante
    sorted_models = sorted(models_test.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_models]
    accuracies_test = [item[1] * 100 for item in sorted_models]
    accuracies_train = [models_train[name] * 100 for name in names]
    
    # Créer la figure avec 2 sous-graphiques
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Graphique 1: Comparaison Train (CV) vs Test ---
    ax1 = axes[0]
    x = np.arange(len(names))
    width = 0.35
    
    bars_train = ax1.bar(x - width/2, accuracies_train, width, label='Train (CV)', color='#3498db', edgecolor='black', linewidth=1)
    bars_test = ax1.bar(x + width/2, accuracies_test, width, label='Test', color='#2ecc71', edgecolor='black', linewidth=1)
    
    # Ligne de baseline
    ax1.axhline(y=baseline * 100, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline*100:.1f}%)')
    
    # Ajouter les valeurs sur les barres
    for bar in bars_train:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{bar.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold', color='#2874A6')
    for bar in bars_test:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{bar.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold', color='#1E8449')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Comparaison Train (CV) vs Test', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.split('(')[0].strip() for n in names], rotation=15, ha='right')
    ax1.set_ylim(55, 75)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)
    
    # --- Graphique 2: Écart Train-Test ---
    ax2 = axes[1]
    ecarts = [train - test for train, test in zip(accuracies_train, accuracies_test)]
    
    # Couleur uniforme bleue pour toutes les barres
    bars2 = ax2.bar(range(len(names)), ecarts, color='#3498db', edgecolor='black', linewidth=1.2)
    
    # Ajouter les valeurs sur les barres
    for i, (bar, ecart) in enumerate(zip(bars2, ecarts)):
        y_pos = bar.get_height() + 0.1 if ecart >= 0 else bar.get_height() - 0.3
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos, 
                f'{ecart:+.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.split('(')[0].strip() for n in names], rotation=15, ha='right')
    ax2.set_ylabel('Écart Train - Test (%)', fontsize=12)
    ax2.set_title('Écart Train vs Test', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('comparaison_train_test.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
  print("=" * 60)
  print("CHARGEMENT DES DONNÉES")
  print("=" * 60)
  games = read_data("csv/game.csv")
  train_games, test_games = split_train_test(games, ratio=0.7)

  # Features dynamiques
  train_X, train_Y = build_features_dynamic(train_games)
  test_X, test_Y = build_features_dynamic(test_games)

  # Normalisation
  train_X_norm, test_X_norm = normalize(train_X, test_X)

  # Baseline
  baseline = sum(test_Y) / len(test_Y)
  print(f"\n  Baseline (toujours domicile): {baseline:.4f}")

  # ============================================================
  # K-NN
  # ============================================================
  print("\n" + "=" * 60)
  print("K-NN")
  print("=" * 60)
  
  def knn_classifier_untrained(tx, ty, k, x):
    return predict_knn(x, tx, ty, distance_euclidienne, k)
  
  # Cross-validation pour trouver le meilleur K
  # best_k, best_error_knn = find_best_k(train_X_norm, train_Y, knn_classifier_untrained)
  best_k, best_error_knn = 228, 0.3178  # Valeur pré-calculée
  print(f"  Meilleur K: {best_k}, accuracy: {1-best_error_knn}")
  
  # Évaluation finale sur TEST
  # knn_classifier = lambda x: predict_knn(x, train_X_norm, train_Y, distance_euclidienne, best_k)
  # knn_error = eval_classifier(test_X_norm, test_Y, knn_classifier)
  # knn_accuracy = 1 - knn_error
  knn_accuracy = 0.6511
  print(f"  Accuracy sur TEST: {knn_accuracy:.4f}")

  # ============================================================
  # SVM
  # ============================================================
  print("\n" + "=" * 60)
  print("SVM")
  print("=" * 60)
  
  # Cross-validation pour trouver le meilleur C
  # best_c_svm, best_error_svm = find_best_c(train_X_norm, train_Y)
  best_c_svm, best_error_svm = 0.1, 0.3148 # Valeur pré-calculée
  print(f"  Meilleur C: {best_c_svm}, accuracy: {1-best_error_svm}")
  
  # Évaluation finale sur TEST
  # svm_predictions = SVM_classifier(train_X_norm, train_Y, test_X_norm, C=best_c_svm)
  # svm_errors = sum(1 for i in range(len(test_Y)) if svm_predictions[i] != test_Y[i])
  # svm_accuracy = 1 - (svm_errors / len(test_Y))
  svm_accuracy = 0.6468
  print(f"  Accuracy sur TEST: {svm_accuracy:.4f}")

  # ============================================================
  # RANDOM FOREST
  # ============================================================
  print("\n" + "=" * 60)
  print("RANDOM FOREST")
  print("=" * 60)
  
  # Cross-validation pour trouver les meilleurs paramètres
  # best_n_est, best_max_depth, best_error = find_best_rf_params(train_X_norm, train_Y)
  best_n_est, best_max_depth, best_error_rf = 300, 10, 0.3148   # Valeurs pré-calculées
  depth_str = str(best_max_depth) if best_max_depth else "None"
  print(f"  Meilleurs params: n_estimators={best_n_est}, max_depth={depth_str}, accuracy={1-best_error_rf}")
  
  # Évaluation finale sur TEST
  rf_predictions, rf_model = RF_classifier(
      train_X_norm, train_Y, test_X_norm,
      n_estimators=best_n_est, max_depth=best_max_depth
  )
  rf_accuracy = accuracy_score(test_Y, rf_predictions)
  print(f"  Accuracy sur TEST: {rf_accuracy:.4f}")
  
  print("\n  Importance des features:")
  for name, imp in sorted(zip(FEATURES_NAME, rf_model.feature_importances_), key=lambda x: x[1], reverse=True)[:5]:
      print(f"    {name:30} {imp:.4f}")

  # ============================================================
  # LOGISTIC REGRESSION
  # ============================================================
  print("\n" + "=" * 60)
  print("LOGISTIC REGRESSION")
  print("=" * 60)
  
  # Cross-validation pour trouver le meilleur C
  # best_c_lr, best_error_lr = find_best_lr_C(train_X_norm, train_Y)
  best_c_lr, best_error_lr = 0.1, 0.3141 # Valeur pré-calculée
  print(f"  Meilleur C: {best_c_lr}, accuracy: {1-best_error_lr}")
  
  # Évaluation finale sur TEST
  lr_predictions, lr_model = LR_classifier(
      train_X_norm, train_Y, test_X_norm, C=best_c_lr
  )
  lr_accuracy = accuracy_score(test_Y, lr_predictions)
  print(f"  Accuracy sur TEST: {lr_accuracy:.4f}")
  
  print("\n  Coefficients (top 5):")
  coefs = list(zip(FEATURES_NAME, lr_model.coef_[0]))
  for name, coef in sorted(coefs, key=lambda x: abs(x[1]), reverse=True)[:5]:
      print(f"    {name:30} {coef:+.4f}")

  # ============================================================
  # RÉSULTATS FINAUX
  # ============================================================
  print("\n" + "=" * 60)
  print("RÉSULTATS FINAUX")
  print("=" * 60)
  
  results_test = {
    'Baseline (domicile)': baseline,
    f'K-NN (K={best_k})': knn_accuracy,
    f'SVM (C={best_c_svm})': svm_accuracy,
    f'Random Forest (n={best_n_est})': rf_accuracy,
    f'Logistic Reg. (C={best_c_lr})': lr_accuracy
  }
  
  results_train = {
    'Baseline (domicile)': baseline,
    f'K-NN (K={best_k})': 1 - best_error_knn,
    f'SVM (C={best_c_svm})': 1 - best_error_svm,
    f'Random Forest (n={best_n_est})': 1 - best_error_rf,
    f'Logistic Reg. (C={best_c_lr})': 1 - best_error_lr
  }
  
  print(f"\n{'Modèle':<35} {'Train (CV)':>12} {'Test':>10} {'Écart':>10}")
  print("-" * 70)
  for name in results_test.keys():
    if name == 'Baseline (domicile)':
      print(f"{name:<35} {'-':>12} {results_test[name]*100:>9.1f}% {'-':>10}")
    else:
      train_acc = results_train[name] * 100
      test_acc = results_test[name] * 100
      ecart = train_acc - test_acc
      print(f"{name:<35} {train_acc:>11.1f}% {test_acc:>9.1f}% {ecart:>+9.1f}%")
  
  print("-" * 70)
  
  models_only = {k: v for k, v in results_test.items() if k != 'Baseline (domicile)'}
  best_model = max(models_only, key=models_only.get)
  best_acc = models_only[best_model]
  
  print(f"\nMeilleur modèle: {best_model}")
  print(f"   Accuracy Test: {best_acc*100:.1f}%")
  print(f"   Amélioration vs baseline: {(best_acc - baseline)*100:+.1f}%")
  
  print("\n Classement (sur Test):")
  sorted_models = sorted(models_only.items(), key=lambda x: x[1], reverse=True)
  for i, (name, acc) in enumerate(sorted_models, 1):
    print(f"   {i}. {name}: {acc*100:.1f}%")
  
  # Graphique comparatif Train vs Test
  plot_all_models_comparison(results_test, results_train, baseline)


if __name__ == "__main__":
  main()
