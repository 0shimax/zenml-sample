from data_processing.pipeline import feature_engineering
from evaluator.pipeline import evaluate
from pipeline_utils.pipeline_config_loader import load_parameters
from predictor.pipeline import inference
from trainer.pipeline import training
from zenml.logger import get_logger

logger = get_logger(__name__)


# default artifact
# https://docs.zenml.io/reference/global-settings
# ~/.config/zenml/local_stores/以下
# cloudはこれで設定
# https://docs.zenml.io/stacks/deployment/register-a-cloud-stack
# https://docs.zenml.io/concepts/stack_components
# localはこれでも設定かのう
# https://docs.zenml.io/concepts/artifacts#register-an-existing-folder
def main():
    """Main entry point for the pipeline execution."""

    parameters = load_parameters()

    # Execute Feature Engineering Pipeline
    feature_engineering.with_options()(
        parameters["data_processing"], parameters["global"]
    )
    logger.info("Feature Engineering pipeline finished successfully!\n")

    # for deploy
    training.with_options()()
    logger.info("Training pipeline successfully!\n\n")

    # Run the pipeline
    inference()
    logger.info("Inference pipeline finished successfully!")

    evaluate()
    logger.info("Evaluation pipeline finished successfully!")


if __name__ == "__main__":
    main()
