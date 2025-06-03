import numpy as np
import pytest


def swap_random_features(sample: np.ndarray, rng: np.random.Generator):
    """
    Given a 1D numpy array `sample`, swap two random indices and return:
    (swapped_array, index_i, index_j).
    """
    n = sample.shape[0]
    i, j = rng.integers(0, n), rng.integers(0, n)
    swapped = sample.copy()
    swapped[i], swapped[j] = swapped[j], swapped[i]
    return swapped, i, j


@pytest.fixture
def test_data(trained_model):
    """
    Provide synthetic test data whose number of features exactly matches
    trained_model.n_features_in_. Returns a dictionary with keys "X" and "y".
    """
    rng = np.random.default_rng(seed=42)
    n_features = trained_model.n_features_in_
    X = rng.random((10, n_features))
    y = rng.integers(0, 2, size=(10,))
    return {"X": X, "y": y}


def test_prediction_stability_under_feature_swap(trained_model, test_data):
    """
    Metamorphic test: swapping two random features in any test sample
    should not change the GaussianNB’s predicted class.
    """
    X = test_data["X"]
    num_samples = X.shape[0]
    rng = np.random.default_rng(seed=42)

    for _ in range(2):
        sample_idx = rng.integers(0, num_samples)
        original_sample = X[sample_idx].copy()
        modified_sample, i, j = swap_random_features(original_sample, rng)

        pred_orig = trained_model.predict([original_sample])[0]
        pred_mod = trained_model.predict([modified_sample])[0]

        assert pred_orig == pred_mod, (
            f"Prediction changed after swapping features {i} and {j} "
            f"in sample {sample_idx}: {pred_orig} → {pred_mod}"
        )