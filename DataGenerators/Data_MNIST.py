"""
    Generator for the MNIST Dataset.
"""

import numpy as np
from sklearn.datasets import fetch_openml

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_global, y_global = mnist.data / 255.0, mnist.target.astype(int)

# Split all digits into two parts. 
def get_mnist_digit_groups(use_odd_even_split=True, random_seed=42, class_count=5):
    if use_odd_even_split:
        class1_digits = np.array([0, 2, 4, 6, 8])
        class2_digits = np.array([1, 3, 5, 7, 9])
    else:
        rng = np.random.default_rng(random_seed)
        all_digits = np.arange(10)
        rng.shuffle(all_digits)
        class1_digits = all_digits[:class_count]
        class2_digits = all_digits[class_count: 2 * class_count]

    return class1_digits, class2_digits

# Get traning data. 
def generate_MNIST_train(N_train, class1_digits, class2_digits):
    """Sample N_train images for each digit in each digit group."""
    
    # Training data pool: 0-10000
    X_train_pool = X_global[:10000]
    y_train_pool = y_global[:10000]

    def sample_digit_balanced_from_pool(digits, n_samples_per_digit):
        grouped_samples = []
        for digit in np.asarray(digits, dtype=int):
            idx = np.where(y_train_pool == digit)[0]
            if len(idx) < n_samples_per_digit:
                raise ValueError(f"No enough training samples for digit {digit}!")

            np.random.shuffle(idx)
            grouped_samples.append(X_train_pool[idx[:n_samples_per_digit]])

        return np.vstack(grouped_samples)

    X_train_1 = sample_digit_balanced_from_pool(class1_digits, N_train)
    X_train_2 = sample_digit_balanced_from_pool(class2_digits, N_train)
    
    return X_train_1, X_train_2

# Get testing data with grouped observations.
def generate_MNIST_test_class_observations(
    N_test,
    observation_size,
    class1_digits,
    class2_digits,
    random_seed=42,
    return_digit_labels=False,
):
    # Each observation batch belongs to one digit class group, but the images
    # inside the batch are not forced to be the same concrete digit.
    X_test_pool = X_global[10000:20000]
    y_test_pool = y_global[10000:20000]
    rng = np.random.default_rng(random_seed)

    def sample_grouped_from_pool(digits):
        digits = np.asarray(digits, dtype=int)
        grouped_samples = []
        grouped_labels = []

        digit_indices = {}
        digit_offsets = {}
        for digit in digits:
            idx = np.where(y_test_pool == digit)[0]
            if len(idx) == 0:
                raise ValueError(f"No testing samples for digit {digit}!")
            digit_indices[int(digit)] = rng.permutation(idx)
            digit_offsets[int(digit)] = 0

        def draw_one(digit):
            digit = int(digit)
            idx = digit_indices[digit]
            offset = digit_offsets[digit]
            if offset >= len(idx):
                idx = rng.permutation(np.where(y_test_pool == digit)[0])
                digit_indices[digit] = idx
                offset = 0
            selected = idx[offset]
            digit_offsets[digit] = offset + 1
            return selected

        for _ in range(N_test):
            if observation_size > 1 and len(digits) > 1:
                first_digit = rng.choice(digits)
                alternative_digits = digits[digits != first_digit]
                second_digit = rng.choice(alternative_digits)
                remaining_digits = rng.choice(digits, size=observation_size - 2, replace=True)
                batch_digits = np.concatenate(
                    [np.array([first_digit, second_digit]), remaining_digits]
                )
            else:
                batch_digits = rng.choice(digits, size=observation_size, replace=True)

            selected = [draw_one(digit) for digit in batch_digits]
            grouped_samples.append(X_test_pool[selected])
            grouped_labels.extend(batch_digits.tolist())

        X_test = np.vstack(grouped_samples)
        y_test = np.array(grouped_labels)

        return X_test, y_test

    X_test_1, y_test_1 = sample_grouped_from_pool(class1_digits)
    X_test_2, y_test_2 = sample_grouped_from_pool(class2_digits)

    if return_digit_labels:
        return X_test_1, X_test_2, y_test_1, y_test_2

    return X_test_1, X_test_2

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
