"""
    Generator for the Higgs Dataset.
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd

# The CSV file is saved locally in this path. 
# You can download the HIGGS dataset from https://archive.ics.uci.edu/ml/datasets/HIGGS and save it as a CSV file.

DEFAULT_HIGGS_PATH = r"E:\Datasets\higgs\HIGGS.csv"
DEFAULT_POOL_SIZE_PER_CLASS = 300000


# Load Higgs dataset from a CSV file and return two class pools of specified size.
# The first CSV column is the binary label, and columns 2-22 are used as the 21 features requested for this experiment. 
@lru_cache(maxsize=2)
def load_higgs_class_pools(
    data_path=DEFAULT_HIGGS_PATH,
    pool_size_per_class=DEFAULT_POOL_SIZE_PER_CLASS,
    chunksize=200000,
):

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"HIGGS data file not found: {data_path}")

    class0_chunks = []
    class1_chunks = []
    class0_count = 0
    class1_count = 0

    for chunk in pd.read_csv(data_path, header=None, usecols=list(range(22)), chunksize=chunksize):
        y_chunk = chunk.iloc[:, 0].to_numpy(dtype=np.int64)
        X_chunk = chunk.iloc[:, 1:22].to_numpy(dtype=np.float64)

        if class0_count < pool_size_per_class:
            X0 = X_chunk[y_chunk == 0]
            if len(X0):
                take = min(pool_size_per_class - class0_count, len(X0))
                class0_chunks.append(X0[:take])
                class0_count += take

        if class1_count < pool_size_per_class:
            X1 = X_chunk[y_chunk == 1]
            if len(X1):
                take = min(pool_size_per_class - class1_count, len(X1))
                class1_chunks.append(X1[:take])
                class1_count += take

        if class0_count >= pool_size_per_class and class1_count >= pool_size_per_class:
            break

    if class0_count == 0 or class1_count == 0:
        raise ValueError("HIGGS data must contain both labels 0 and 1.")

    return np.vstack(class0_chunks), np.vstack(class1_chunks)

# Generate one normalized HIGGS train/test split with common observation panels. 
def get_higgs_train_test(X1_pool, X2_pool, N_train, max_train, N_test, max_observation_size, trial):
    rng = np.random.default_rng(trial)
    idx1 = rng.permutation(len(X1_pool))
    idx2 = rng.permutation(len(X2_pool))

    n_test_rows = N_test * max_observation_size
    required = max_train + n_test_rows
    if len(idx1) < required or len(idx2) < required:
        raise ValueError(f"Need {required} samples per class for common HIGGS split.")

    X1 = X1_pool[idx1[:N_train]]
    X2 = X2_pool[idx2[:N_train]]
    X_test_1_panel = X1_pool[idx1[max_train:required]]
    X_test_2_panel = X2_pool[idx2[max_train:required]]

    mean_all = np.concatenate((X1, X2), axis=0).mean(axis=0)
    std_all = np.concatenate((X1, X2), axis=0).std(axis=0) + 1e-8
    return (
        (X1 - mean_all) / std_all,
        (X2 - mean_all) / std_all,
        (X_test_1_panel - mean_all) / std_all,
        (X_test_2_panel - mean_all) / std_all,
    )

# Generate largest tesing groups. 
def select_observation_prefix(X_panel, observation_size, max_observation_size):
    if observation_size > max_observation_size:
        raise ValueError("observation_size cannot exceed max_observation_size.")
    if X_panel.shape[0] % max_observation_size != 0:
        raise ValueError("X_panel rows must be divisible by max_observation_size.")

    n_groups, d = X_panel.shape[0] // max_observation_size, X_panel.shape[1]
    return X_panel.reshape(n_groups, max_observation_size, d)[:, :observation_size, :].reshape(
        n_groups * observation_size, d
    )
