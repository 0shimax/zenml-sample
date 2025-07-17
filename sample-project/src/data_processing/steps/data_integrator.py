import polars as pl
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def merge_data(
    preprocessed_companies: pl.DataFrame,
    raw_reviews: pl.DataFrame,
    preprocessed_shuttles: pl.DataFrame,
) -> Annotated[pl.DataFrame, "merged_dataset"]:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    rated_shuttles = preprocessed_shuttles.join(
        raw_reviews, left_on="id", right_on="shuttle_id"
    )
    rated_shuttles = rated_shuttles.drop("id")
    model_input_table = rated_shuttles.join(
        preprocessed_companies, left_on="company_id", right_on="id"
    )
    model_input_table = model_input_table.drop_nulls()
    return model_input_table
