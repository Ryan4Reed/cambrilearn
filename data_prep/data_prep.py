import pandas as pd
import logging
import os

reference_date = pd.Timestamp("2024-07-01")


def setup_logging():
    """
    Setup logger.
    """
    # Ensure logs folder exists
    os.makedirs("logs", exist_ok=True)

    # Create a logger specific for this file
    logger = logging.getLogger("data_prep")
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler("logs/data_prep.log", mode="w")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add handler if not already added
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
    return logger


logger = setup_logging()


def drop_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Drop columns that will not form part of analysis.
    """
    data = data.drop(columns=columns)
    logger.info(f"Successfully dropped the following columns: {columns}")
    return data


def convert_entries_to_nan(
    data: pd.DataFrame, column: str, entries: list[str]
) -> pd.DataFrame:
    """
    Converts entries for a particular column to NaN based on value match.
    """
    affected_count = data[column].isin(entries).sum()
    data[column] = data[column].replace(entries, None)
    logger.info(
        f"Converted {affected_count} entries matching: {entries} to NaN from column: {column}"
    )
    return data


def convert_to_datetime(data: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    """
    Converts columns to pandas datetime object
    """
    data[date_columns] = data[date_columns].apply(pd.to_datetime)
    logger.info(f"Converted the following columns to datetime: {date_columns}")
    return data


def drop_rows_outside_period(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all rows for which date_received is on and after cutoff_date.
    """
    init_count = len(data)
    data = data[data["date_received"] < reference_date]
    logger.info(
        f"Dropped {init_count - len(data)} rows with date_received on/after {reference_date}"
    )
    return data


def drop_rows_with_nan(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Drop rows based on NaN entries wrt a particular set of columns.
    """
    init_count = len(data)
    data = data.dropna(subset=columns)
    logger.info(f"Dropped {init_count - len(data)} rows with NaN in columns: {columns}")
    return data


def recreate_months_since_joined(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Recreate the months_since_joined column based on up to date data.
    """
    data = drop_columns(data=data, columns=[col_name])
    data[col_name] = (2024 - data["start_date"].dt.year) * 12 + (
        12 - data["start_date"].dt.month
    )
    logger.info(f"Successfully created column: {col_name}")
    return data


def recreate_last_year_claim_count(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Recreate last_year_claim_count column based on up to date data.
    """
    data = drop_columns(data=data, columns=[col_name])
    cutoff_date = "2023-06-01"
    counts = data[data["date_received"] >= cutoff_date].groupby("account_id").size()
    # map counts back to df
    data["last_year_claim_count"] = data["account_id"].map(counts).fillna(0).astype(int)
    logger.info(f"Successfully created column: {col_name}")
    return data


def recreate_months_since_last_claim(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Recreate months_since_last_claim column based on up to date data.
    """
    data = drop_columns(data=data, columns=[col_name])
    latest_claim_date = data.groupby("account_id")["date_received"].max()

    months_diff = (reference_date.year - latest_claim_date.dt.year) * 12 + (
        reference_date.month - latest_claim_date.dt.month
    )
    data["months_since_last_claim"] = data["account_id"].map(months_diff).astype(int)
    logger.info(f"Successfully created column: {col_name}")
    return data


def recreate_dry_months(data: pd.DataFrame, col_names: str) -> pd.DataFrame:
    """
    Recreate dry_months column based on up to date data.
    Also adding four_years_dry column.
    """
    data = drop_columns(data=data, columns=col_names)
    mapping = {
        "two_months_dry": 2,
        "six_months_dry": 6,
        "one_year_dry": 12,
        "two_years_dry": 24,
        "four_years_dry": 48,
    }
    for column in col_names + ["four_years_dry"]:
        data[column] = (data["months_since_last_claim"] > mapping[column]).astype(int)

    logger.info(f"Successfully created columns: {col_names}")
    pd.set_option("display.max_columns", 50)
    logger.info(data)
    return data


def clean_industry_segment(data: pd.DataFrame) -> pd.DataFrame:
    """
    Update NaN industry_segment entries if account_id has valid value in another row.
    One to one relationship between industry and account
    """
    missing_before = data["industry_segment"].isna().sum()
    # Get mapping of account_id to first valid industry
    industry_mapping = (
        data.dropna(subset=["industry_segment"])
        .groupby("account_id")["industry_segment"]
        .first()
    )
    # Fill missing or unassigned values by mapping account_id
    data["industry_segment"] = data["industry_segment"].fillna(
        data["account_id"].map(industry_mapping)
    )
    missing_after = data["industry_segment"].isna().sum()
    logger.info(
        f"Was able to correct a total of {missing_before - missing_after} entries wrt industry_segment"
    )
    logger.info(
        f"There is still a total of {data['industry_segment'].isna().sum()} NaN entries in industry_segment"
    )
    return data


def prep_claims_data() -> pd.DataFrame:
    data = pd.read_csv("data/all_claims_data.csv")
    # drop irrelevant columns
    data = drop_columns(
        data=data, columns=["case_number", "is_closed", "is_deleted", "type"]
    )

    # Convert unknown account_id entries to NaN
    data = convert_entries_to_nan(
        data=data, column="account_id", entries=["Unknown Account"]
    )
    data = drop_rows_with_nan(
        data=data, columns=["account_id"]
    )  # No such entries effect expected profit

    # convert illogical loss_date entries to NaN
    data = convert_entries_to_nan(
        data=data, column="loss_date", entries=["3120-04-03", "3130-04-03"]
    )
    data = convert_to_datetime(
        data=data, date_columns=["date_received", "loss_date", "start_date"]
    )
    data = drop_rows_outside_period(data=data)
    data = recreate_months_since_joined(data=data, col_name="months_since_joined")
    data = recreate_last_year_claim_count(data=data, col_name="last_year_claim_count")
    data = recreate_months_since_last_claim(
        data=data, col_name="months_since_last_claim"
    )
    data = recreate_dry_months(
        data=data,
        col_names=["two_months_dry", "six_months_dry", "one_year_dry", "two_years_dry"],
    )
    # drop date columns now that we no longer need them
    data = drop_columns(data=data, columns=["date_received", "loss_date", "start_date"])

    # Convert Unassigned industry_segment entries to NaN
    data = convert_entries_to_nan(
        data=data, column="industry_segment", entries=["Unassigned"]
    )
    data = clean_industry_segment(data=data)


if __name__ == "__main__":
    try:
        prep_claims_data()

    except Exception as error:
        logger.error(error)
        raise error
