import numpy as np
import pytest

def swap_random_features(sample, rng):
    """Swap two randomly chosen features in the input sample."""
    x_swapped = sample.copy()
    n_features = x_swapped.shape[0]
    i, j = rng.choice(n_features, size=2, replace=False)
    x_swapped[i], x_swapped[j] = x_swapped[j], x_swapped[i]
    return x_swapped, i, j

def test_prediction_stability_under_feature_swap(trained_model, test_data):
    """
    Metamorphic test: Swapping two random features in test samples should not affect the predicted class.
    """
    X = test_data["X"]
    num_samples = X.shape[0]
    rng = np.random.default_rng(seed=42)

    for _ in range(2):
        sample_idx = rng.integers(0, num_samples)
        original_sample = X[sample_idx].copy()
        modified_sample, i, j = swap_random_features(original_sample, rng)

        prediction_original = trained_model.predict([original_sample])[0]
        prediction_modified = trained_model.predict([modified_sample])[0]

        assert prediction_original == prediction_modified, (
            f"Prediction changed after swapping features {i} and {j} "
            f"in sample index {sample_idx}: {prediction_original} -> {prediction_modified}"
        )
    p