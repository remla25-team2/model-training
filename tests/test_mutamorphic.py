import numpy as np
import pytest

def swap_random_features(sample, rng):
    """Swap two randomly chosen features in the input sample."""
    x_swapped = sample.copy()
    n_features = x_swapped.shape[0]
    i, j = rng.choice(n_features, size=2, replace=False)
    x_swapped[i], x_swapped[j] = x_swapped[j], x_swapped[i]
    return x_swapped, i, j

@pytest.fixture
def test_data():
    """
    Provide synthetic test data with exactly 13 features per sample,
    matching what the trained_model expects.
    """
    rng = np.random.default_rng(seed=42)
    X = rng.random((10, 13))
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