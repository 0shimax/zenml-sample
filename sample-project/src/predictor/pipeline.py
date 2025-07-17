from zenml import pipeline
from zenml.client import Client
from zenml.logger import get_logger

from .steps.predictor import predict

logger = get_logger(__name__)
client = Client()


@pipeline
def inference():
    """
    Model inference pipeline.
    """
    # Get the production model artifact
    # model = get_pipeline_context().model.get_artifact("lgbm_regressor_eval")
    # model = client.get_model_version(model_name_or_id="lgbm_regressor_eval")
    model = client.get_artifact_version(name_id_or_prefix="lgbm_regressor_eval")
    test_features = client.get_artifact_version(name_id_or_prefix="test_features")

    predict(
        model,
        test_features,
    )
