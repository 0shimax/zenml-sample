from zenml import pipeline
from zenml.client import Client
from zenml.logger import get_logger

from .steps.model_evaluator import evaluate_model

logger = get_logger(__name__)


@pipeline
def evaluate():
    """
    Model training pipeline.
    """
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.
    client = Client()

    # model = client.get_artifact_version("lgbm_regressor")
    eval_model = client.get_artifact_version("lgbm_regressor_eval")
    test_features = client.get_artifact_version(name_id_or_prefix="test_features")
    test_targets = client.get_artifact_version(name_id_or_prefix="test_targets")

    score = evaluate_model(
        model=eval_model,
        test_features=test_features,
        test_targets=test_targets,
    )

    # promote_model(score=score)
