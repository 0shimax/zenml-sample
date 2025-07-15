import polars as pl
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from typing_extensions import Annotated
from zenml import ArtifactConfig, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_trainer(
    features: pl.DataFrame,
    targets: pl.DataFrame,
) -> Annotated[
    ClassifierMixin,
    ArtifactConfig(name="sklearn_classifier", is_model_artifact=True),
]:
    """Configure and train a model on the training dataset.

    This is an example of a model training step that takes in a dataset artifact
    previously loaded and pre-processed by other steps in your pipeline, then
    configures and trains a model on it. The model is then returned as a step
    output artifact.

    Args:
        dataset_trn: The preprocessed train dataset.
        model_type: The type of model to train.
        target: The name of the target column in the dataset.

    Returns:
        The trained model artifact.

    Raises:
        ValueError: If the model type is not supported.
    """
    # Initialize the model with the hyperparameters indicated in the step
    # parameters and train it on the training set.
    model = RandomForestClassifier()
    logger.info(f"Training model {model}...")

    model.fit(
        features,
        targets,
    )
    return model
