from typing import Tuple

import polars as pl
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from zenml import step


@step
def split_data_into_features_and_targets(
    dataset: pl.DataFrame, parameters: dict
) -> Tuple[
    Annotated[pl.DataFrame, "features"],
    Annotated[pl.DataFrame, "targets"],
]:
    features = dataset.select(pl.col(parameters["features"]))
    targets = dataset.select(pl.col(parameters["target_colmn"]))
    return features, targets


@step
def split_data_for_train_and_test(
    features: pl.DataFrame, targets: pl.DataFrame, parameters: dict
) -> Tuple[
    Annotated[pl.DataFrame, "train_features"],
    Annotated[pl.DataFrame, "test_features"],
    Annotated[pl.DataFrame, "train_targets"],
    Annotated[pl.DataFrame, "test_targets"],
]:
    """Dataset splitter step."""
    train_features, test_features, train_targets, test_targets = train_test_split(
        features,
        targets,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        shuffle=parameters["shuffle"],
    )
    return train_features, test_features, train_targets, test_targets
