from data_processing.pipeline import feature_engineering
from pipeline_utils.pipeline_config_loader import load_parameters
from zenml.logger import get_logger

logger = get_logger(__name__)
parameters = load_parameters()


def main(parameters: dict):
    """Main entry point for the pipeline execution."""
    # client = Client()

    # Execute Feature Engineering Pipeline
    # feature_engineering.with_options()(
    #     parameters["data_processing"], parameters["global"]
    # )
    feature_engineering(parameters["data_processing"], parameters["global"])
    logger.info("Feature Engineering pipeline finished successfully!\n")

    # train_dataset_artifact = client.get_artifact_version(train_dataset_name)
    # test_dataset_artifact = client.get_artifact_version(test_dataset_name)
    # logger.info(
    #     "The latest feature engineering pipeline produced the following "
    #     f"artifacts: \n\n1. Train Dataset - Name: {train_dataset_name}, "
    #     f"Version Name: {train_dataset_artifact.version} \n2. Test Dataset: "
    #     f"Name: {test_dataset_name}, Version Name: {test_dataset_artifact.version}"
    # )

    # Execute Training Pipeline
    # # If train_dataset_version_name is specified, use versioned artifacts
    # if train_dataset_version_name or test_dataset_version_name:
    #     # However, both train and test dataset versions must be specified
    #     assert (
    #         train_dataset_version_name is not None
    #         and test_dataset_version_name is not None
    #     )
    #     train_dataset_artifact_version = client.get_artifact_version(
    #         train_dataset_name, train_dataset_version_name
    #     )
    #     # If train dataset is specified, test dataset must be specified
    #     test_dataset_artifact_version = client.get_artifact_version(
    #         test_dataset_name, test_dataset_version_name
    #     )
    #     # Use versioned artifacts
    #     run_args_train["train_dataset_id"] = train_dataset_artifact_version.id
    #     run_args_train["test_dataset_id"] = test_dataset_artifact_version.id

    # # Run the RF pipeline
    # # pipeline_args = {}
    # # pipeline_args["config_path"] = os.path.join(config_folder, "training_rf.yaml")
    # training.with_options(**parameters["training"]["settings"])(**["training"]["run"])
    # logger.info("Training pipeline with RF finished successfully!\n\n")

    # # Configure the pipeline
    # inference_configured = inference.with_options(
    #     **parameters["prediction"]["settings"]
    # )

    # # zenml_model = client.get_model_version(
    # #     parameters["model"]["name"], parameters["model"]["version"]
    # # )
    # # preprocess_pipeline_artifact = zenml_model.get_artifact("preprocess_pipeline")

    # # # Use the metadata of feature engineering pipeline artifact
    # # #  to get the random state and target column
    # # random_state = preprocess_pipeline_artifact.run_metadata["random_state"]
    # # target = preprocess_pipeline_artifact.run_metadata["target"]
    # # run_args_inference["random_state"] = random_state
    # # run_args_inference["target"] = target

    # # Run the pipeline
    # inference_configured(**parameters["prediction"]["run"])
    # logger.info("Inference pipeline finished successfully!")


if __name__ == "__main__":
    main(parameters)
