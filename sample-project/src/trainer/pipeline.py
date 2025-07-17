from zenml import pipeline
from zenml.client import Client
from zenml.logger import get_logger

from .steps.trainer import train_model

logger = get_logger(__name__)
client = Client()


@pipeline
def training():
    """Model training pipeline."""
    features = client.get_artifact_version(name_id_or_prefix="features")
    targets = client.get_artifact_version(name_id_or_prefix="targets")
    _ = train_model.with_options(substitutions={"model_suffix": ""})(features, targets)

    features = client.get_artifact_version(name_id_or_prefix="train_features")
    targets = client.get_artifact_version(name_id_or_prefix="train_targets")
    _ = train_model.with_options(substitutions={"model_suffix": "_eval"})(
        features, targets
    )
