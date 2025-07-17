from zenml import pipeline
from zenml.logger import get_logger

from .steps import (
    data_integrator,
    data_loader,
    data_preprocessor,
    data_splitter,
)

logger = get_logger(__name__)


@pipeline
def feature_engineering(
    steps_parameters: dict,
    global_parameters: dict,
):
    """
    Feature engineering pipeline.
    """
    raw_companies = data_loader.load_companies(
        steps_parameters["data_loader"]["companies"], global_parameters
    )
    raw_reviews = data_loader.load_reviews(
        steps_parameters["data_loader"]["reviews"], global_parameters
    )
    raw_shuttles = data_loader.load_shuttles(
        steps_parameters["data_loader"]["shuttles"], global_parameters
    )

    processed_companies = data_preprocessor.preprocess_companies(raw_companies)
    processed_shuttles = data_preprocessor.preprocess_shuttles(raw_shuttles)

    merged_data = data_integrator.merge_data(
        processed_companies,
        raw_reviews,
        processed_shuttles,
    )

    features, targets = data_splitter.split_data_into_features_and_targets(
        dataset=merged_data,
        parameters=steps_parameters["data_splitter"]["features_and_targets"],
    )
    train_features, test_features, train_targets, test_targets = (
        data_splitter.split_data_for_train_and_test(
            features, targets, steps_parameters["data_splitter"]["train_and_test"]
        )
    )
    return train_features, test_features, train_targets, test_targets
