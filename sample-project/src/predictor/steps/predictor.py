import lightgbm as lgbm
import polars as pl
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def predict(
    model: lgbm.LGBMRegressor,
    features: pl.DataFrame,
) -> Annotated[pl.Series, "predictions"]:
    """Predictions step."""
    # run prediction from memory
    predictions = model.predict(features)

    predictions = pl.Series("predicted", predictions)
    return predictions
