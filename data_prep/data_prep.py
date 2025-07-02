import pandas as pd
import logging
import os


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
    file_handler = logging.FileHandler("logs/data_prep.log")
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


def convert_illogical_dates_nan(
    data: pd.DataFrame, column: str, dates: list[str]
) -> pd.DataFrame:
    """
    Converts entries for a particular column to NaN based on value match.
    We motivate in our documentation that illogical dates will be treated as if they are missing
    """
    affected_count = data[column].isin(dates).sum()
    data[column] = data[column].replace(dates, None)
    logger.info(f"Converted {affected_count} entries to NaN from {column}: {dates}")
    return data


def convert_to_datetime(data: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    """
    Converts columns to pandas datetime object
    """
    data[date_columns] = data[date_columns].apply(pd.to_datetime)
    logger.info(f"Converted the following columns to datetime: {date_columns}")
    return data


def drop_rows_outside_period(data: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """
    Drop all rows for which date_received is on and after cutoff_date.
    """
    init_count = len(data)
    data = data[data["date_received"] < cutoff_date]
    logger.info(
        f"Dropped {init_count - len(data)} rows with date_received on/after {cutoff_date}"
    )
    return data


def recreate_months_since_joined(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    data = drop_columns(data=data, columns=[col_name])
    data[col_name] = (2024 - data["start_date"].dt.year) * 12 + (
        12 - data["start_date"].dt.month
    )
    logger.info(f"Successfully created column: {col_name}")
    logger.info(data)
    return data


if __name__ == "__main__":
    setup_logging()
    claims_data = pd.read_csv("data/all_claims_data.csv")
    # drop irrelevant columns
    claims_data = drop_columns(
        data=claims_data, columns=["case_number", "is_closed", "is_deleted", "type"]
    )
    claims_data = convert_illogical_dates_nan(
        data=claims_data, column="loss_date", dates=["3120-04-03", "3130-04-03"]
    )
    claims_data = convert_to_datetime(
        data=claims_data, date_columns=["date_received", "loss_date", "start_date"]
    )
    claims_data = drop_rows_outside_period(data=claims_data, cutoff_date="2024-07-01")
    claims_data = recreate_months_since_joined(
        data=claims_data, col_name="months_since_joined"
    )
