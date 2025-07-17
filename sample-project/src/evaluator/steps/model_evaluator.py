import lightgbm as lgbm
import polars as pl
from zenml import log_metadata, step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def evaluate_model(
    model: lgbm.LGBMRegressor,
    test_features: pl.DataFrame,
    test_targets: pl.DataFrame,
    min_test_accuracy: float = 0.0,
    model_name: str = "lgbm_regressor",
) -> float:
    """Evaluate a trained model."""
    # Calculate the model accuracy on the train and test set
    test_score = model.score(
        test_features,
        test_targets,
    )
    logger.info(f"Test accuracy={test_score:.2f}%")

    messages = []
    if test_score < min_test_accuracy:
        messages.append(
            f"Test accuracy {test_score:.2f}% is below {min_test_accuracy:.2f}% !"
        )
    else:
        for message in messages:
            logger.warning(message)

    client = Client()
    latest_classifier = client.get_artifact_version(model_name)

    log_metadata(
        metadata={
            "test_score": float(test_score),
        },
        artifact_version_id=latest_classifier.id,
    )

    return float(test_score)
