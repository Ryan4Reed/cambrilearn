import pandas as pd

from utils.logger import get_logger
from utils.utils import PROPORTION_PRED_COLUMNS, AMOUNT_PRED_COL, SUCCESS_PRED_COL


logger = get_logger("revenue", "logs/revenue.log")

def num_claims(acc_level_data: pd.DataFrame, claim_level_data: pd.DataFrame) -> pd.DataFrame:
    """
    Determine the number of open claims per account.
    """
    num_claims_per_account = (
        claim_level_data[claim_level_data["is_closed"] == False]
        .groupby("account_id")
        .size()
        .reset_index(name="claim_count")
    )
    
    # merge claim_count into data
    acc_level_data = acc_level_data.merge(
        num_claims_per_account, on="account_id", how="left"
    )
    acc_level_data['claim_count'] = acc_level_data['claim_count'].fillna(0).astype(int)

    logger.info("Number of claims per account determined successfully")
    return acc_level_data

def determine_revenue(acc_level_data: pd.DataFrame, claim_level_data: pd.DataFrame):
    """
    Determine expected revenue post 2024-06-30.
    """
    print(">Determining Revenue")
    logger.info(f"\n================== Determine Revenue ==================")
    # get number of claims not closed per account
    acc_level_data = num_claims(acc_level_data=acc_level_data, claim_level_data=claim_level_data)

    # calculate expected amount per account
    acc_level_data["expected_amount"] = (
        acc_level_data["claim_count"] * acc_level_data[AMOUNT_PRED_COL]
    )

    # Adjust expected amount by success rate
    acc_level_data["expected_amount"] = (
        acc_level_data["expected_amount"] * acc_level_data[SUCCESS_PRED_COL]
    )

    # determine final expected amount per month per account
    res_cols = []
    for i, column_name in enumerate(PROPORTION_PRED_COLUMNS, start=1):
        res_col = f"account_rev_month_{i}"
        acc_level_data[res_col] = acc_level_data["expected_amount"] * acc_level_data[column_name]
        res_cols.append(res_col)
    # write data to file
    acc_level_data[['account_id']+ res_cols].to_csv('data/revenue/revenue.csv', index=False)
    logger.info('Successfully wrote revenue data to file')
    logger.info(logger.info(f"\n================== Done =================="))
    print(">Revenue Determined Successfully")
