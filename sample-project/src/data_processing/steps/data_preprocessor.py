import polars as pl
from typing_extensions import Annotated
from zenml import step

# def _is_true(x: pl.Series) -> pl.Series:
#     return x == "t"


# def _parse_percentage(x: pl.Series) -> pl.Series:
#     x = x.str.replace("%", "")
#     x = x.astype(float) / 100
#     return x


# def _parse_money(x: pl.Series) -> pl.Series:
#     x = x.str.replace("$", "").str.replace(",", "")
#     x = x.astype(float)
#     return x


@step
def preprocess_companies(
    companies: pl.DataFrame,
) -> Annotated[pl.DataFrame, "preprocessed_companies"]:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    companies_ch_iata_approved = companies.with_columns(
        bool_iata_approved=(pl.col("iata_approved") == "t")
    )
    companies_ch_company_rating = companies_ch_iata_approved.with_columns(
        cleaned_company_rating=(
            pl.col("company_rating").str.replace("%", "").cast(pl.Float64) / 100
        )
    )
    return companies_ch_company_rating


@step
def preprocess_shuttles(
    shuttles: pl.DataFrame,
) -> Annotated[pl.DataFrame, "preprocessed_shuttles"]:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles_ch_d_check_complete = shuttles.with_columns(
        bool_d_check_complete=(pl.col("d_check_complete") == "t")
    )
    shuttles_ch_moon_clearance_complete = shuttles_ch_d_check_complete.with_columns(
        bool_moon_clearance_complete=(pl.col("moon_clearance_complete") == "t")
    )
    shuttles_ch_price = shuttles_ch_moon_clearance_complete.with_columns(
        float_price=(
            pl.col("price").str.replace("\$", "").str.replace(",", "").cast(pl.Float64)
        )
    )
    return shuttles_ch_price
