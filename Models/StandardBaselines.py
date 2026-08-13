"""
    This file includes all selected standard baseline models for comparison with the Sinkhorn DRO solver. 
"""

import os
import warnings

os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Train a baseline model. 
def trainBaseline(
        base,
        X1_train,
        X2_train,
        hidden_layer_sizes_3nn=(32, 32, 32),
        max_iter=1000,
        tol=1e-3):

    # Construct training dataset with labels
    X_train = np.concatenate([X1_train, X2_train], axis=0)
    y_train = np.concatenate([np.zeros(len(X1_train)), np.ones(len(X2_train))])

    if base == 'GMM':
        # (1) GMM Model
        gmm0 = GaussianMixture(n_components=1, max_iter=max_iter, tol=tol).fit(X1_train)
        gmm1 = GaussianMixture(n_components=1, max_iter=max_iter, tol=tol).fit(X2_train)
        model = (gmm0, gmm1)

    elif base == 'LR':
        # (2) Logistic Regression
        model = LogisticRegression(max_iter=max_iter, tol=tol).fit(X_train, y_train)

    elif base == 'SVM':
        # (3) Kernel SVM
        model = SVC(kernel='rbf', gamma='auto', max_iter=max_iter, tol=tol).fit(X_train, y_train)

    elif base == '3NN':
        # (4) 3-Layer NN
        model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes_3nn),
            max_iter=max_iter,
            tol=tol,
            n_iter_no_change=1,
        ).fit(X_train, y_train)
    
    else:
        raise ValueError(f"Unknown baseline: {base}")

    return model

# Test a baseline model (metric: error rate)
def testBaseline(base, model, X_test_1, X_test_2, observation_size):

    # Construct testing dataset with labels    
    X_test = np.concatenate([X_test_1, X_test_2], axis=0)

    if base == 'GMM':
        # GMM Model
        gmm0 = model[0]
        gmm1 = model[1]
        scores_0 = gmm0.score_samples(X_test)
        scores_1 = gmm1.score_samples(X_test)
        preds = np.where(scores_0 >= scores_1, 0, 1)

    elif base in ['LR', 'SVM', '3NN']:
        preds = model.predict(X_test)
    
    else:
        raise ValueError(f"Unknown baseline: {base}")
    
    # Majority Rule 
    preds_grouped = preds.reshape(-1, observation_size)
    group_votes = preds_grouped.mean(axis=1)
    group_preds = np.where(group_votes > 0.5, 1, 0)

    # Compute total risk
    N_test_0 = len(X_test_1) // observation_size
    N_test_1 = len(X_test_2) // observation_size
    
    y_test_grouped = np.concatenate([np.zeros(N_test_0), np.ones(N_test_1)])

    risk = 1 - accuracy_score(y_test_grouped, group_preds)

    return risk

# Test a baseline model (metrics: error rate, accuracy, f1-score)
def testBaseline_with_metrics(base, model, X_test_1, X_test_2, observation_size):
    X_test = np.concatenate([X_test_1, X_test_2], axis=0)

    if base == 'GMM':
        gmm0 = model[0]
        gmm1 = model[1]
        scores_0 = gmm0.score_samples(X_test)
        scores_1 = gmm1.score_samples(X_test)
        preds = np.where(scores_0 >= scores_1, 0, 1)

    elif base in ['LR', 'SVM', '3NN']:
        preds = model.predict(X_test)

    else:
        raise ValueError(f"Unknown baseline: {base}")

    preds_grouped = preds.reshape(-1, observation_size)
    group_votes = preds_grouped.mean(axis=1)
    group_preds = np.where(group_votes > 0.5, 1, 0)

    N_test_0 = len(X_test_1) // observation_size
    N_test_1 = len(X_test_2) // observation_size
    y_test_grouped = np.concatenate([np.zeros(N_test_0), np.ones(N_test_1)])

    accuracy = accuracy_score(y_test_grouped, group_preds)
    return {
        'error_rate': 1 - accuracy,
        'accuracy': accuracy,
        'f1_score': float(f1_score(y_test_grouped, group_preds, zero_division=0)),
    }
