import pathlib

import polars as pl
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


# ) -> zenml.steps.Output(var_name=pl.DataFrame, var_name2=np.ndarray)
# とかにすればpythonオブジェクトして出力される
@step
def load_companies(
    steps_parameters: dict,
    global_parameters: dict,
) -> Annotated[pl.DataFrame, "dataset"]:
    """Dataset reader step."""
    data_path = pathlib.Path(global_parameters["data_root"]) / pathlib.Path(
        steps_parameters["data_path"]
    )
    dataset = pl.read_csv(data_path)
    logger.info(f"Dataset with {len(dataset)} records loaded!")
    return dataset


@step
def load_reviews(
    steps_parameters: dict,
    global_parameters: dict,
) -> Annotated[pl.DataFrame, "dataset"]:
    """Dataset reader step."""
    data_path = pathlib.Path(global_parameters["data_root"]) / pathlib.Path(
        steps_parameters["data_path"]
    )
    dataset = pl.read_csv(data_path)
    logger.info(f"Dataset with {len(dataset)} records loaded!")
    return dataset


@step
def load_shuttles(
    steps_parameters: dict,
    global_parameters: dict,
) -> Annotated[pl.DataFrame, "dataset"]:
    """Dataset reader step."""
    data_path = pathlib.Path(global_parameters["data_root"]) / pathlib.Path(
        steps_parameters["data_path"]
    )
    dataset = pl.read_excel(data_path)
    logger.info(f"Dataset with {len(dataset)} records loaded!")
    return dataset
