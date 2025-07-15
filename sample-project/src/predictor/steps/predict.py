from typing import Any

import polars as pl
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def inference_predict(
    model: Any,
    dataset_inf: pl.DataFrame,
) -> Annotated[pl.Series, "predictions"]:
    """Predictions step.

    This is an example of a predictions step that takes the data and model in
    and returns predicted values.

    This step is parameterized, which allows you to configure the step
    independently of the step code, before running it in a pipeline.
    In this example, the step can be configured to use different input data.
    See the documentation for more information:

        https://docs.zenml.io/how-to/build-pipelines/use-pipeline-step-parameters

    Args:
        model: Trained model.
        dataset_inf: The inference dataset.

    Returns:
        The predictions as pandas series
    """
    # run prediction from memory
    predictions = model.predict(dataset_inf)

    predictions = pl.Series(predictions, name="predicted")
    return predictions
