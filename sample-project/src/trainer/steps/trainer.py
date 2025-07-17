import lightgbm as lgbm
import polars as pl
from typing_extensions import Annotated
from zenml import ArtifactConfig, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def train_model(
    features: pl.DataFrame,
    targets: pl.DataFrame,
) -> Annotated[
    lgbm.LGBMRegressor,
    ArtifactConfig(name="lgbm_regressor{model_suffix}", is_model_artifact=True),
]:
    """Configure and train a model on the training dataset."""
    model = lgbm.LGBMRegressor()
    logger.info(f"Training model {model}...")

    model.fit(
        features.to_pandas(),
        targets.to_pandas(),
    )
    return model
