import joblib
from training.modeling.model_train import train
from training.modeling.transform import transform
from training.modeling.preprocess import preprocess
from training.modeling.split import split
from sklearn.naive_bayes import GaussianNB
import pickle
import logging
import pytest
from pathlib import Path

# Configure logging for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


@pytest.fixture(scope="module")
def trained_model():
    model_path = Path("models") / "SentimentModel.pkl"
    if not model_path.exists():
        pytest.skip("Trained model not found. Run the pipeline first.")
    model = joblib.load(model_path)
    return model

@pytest.fixture(scope="module")
def bow_vectorizer():
    vectorizer_path = Path("models") / "bow" / "BoW_Sentiment_Model.pkl"
    if not vectorizer_path.exists():
        pytest.skip("Bag-of-Words vectorizer not found. Run the pipeline first.")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer


def test_metamorphic_synonym(trained_model, bow_vectorizer):
    # Ensure necessary artifacts exist before proceeding
    if not isinstance(trained_model, GaussianNB):
        pytest.skip("Trained model not available for metamorphic test.")
    if not hasattr(bow_vectorizer, 'transform'):
        pytest.skip("BoW Vectorizer not available for metamorphic test.")

    original = "This place is sooooo good."
    synonym = "This place is fine."
    
    # Use the TextPreprocessor which should handle the full pipeline
    from lib_ml.preprocessing import TextPreprocessor
    
    # Load the actual preprocessor that was saved during training
    preprocessor_path = Path("models") / "bow" / "BoW_Sentiment_Model.pkl"
    if not preprocessor_path.exists():
        pytest.skip("Preprocessor not found. Run the pipeline first.")
    
    preprocessor = TextPreprocessor.load(preprocessor_path)
    
    # Transform using the loaded preprocessor
    X_orig = preprocessor.transform([original]).toarray()
    X_syn = preprocessor.transform([synonym]).toarray()
    
    pred_orig = trained_model.predict(X_orig)[0]
    pred_syn = trained_model.predict(X_syn)[0]
    
    assert pred_orig in [0, 1]
    assert pred_syn in [0, 1]
    
    # Check if the words 'good' and 'fine' are even in the vocabulary before comparing predictions
    # This avoids false positives if the words aren't recognized by the model
    if (bow_vectorizer.vocabulary_.get("good") is not None and bow_vectorizer.vocabulary_.get("fine") is not None):
        if pred_orig != pred_syn:
            logger.warning(
                f"Metamorphic inconsistency detected: '{original}' (pred={pred_orig}) "
                f"vs '{synonym}' (pred={pred_syn}). "
                "--- Placeholder for Automatic Inconsistency Repair Action ---"
                " e.g., Log this to a monitoring system, add to a re-training dataset, or alert."
            )
            pytest.fail(f"Metamorphic inconsistency: predictions changed for synonym.")
        else:
            logger.info(f"Metamorphic test passed: predictions consistent for synonym ('{original}' and '{synonym}').")

