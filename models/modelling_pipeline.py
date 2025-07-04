import pandas as pd

from utils.utils import PROPORTION_PRED_COLUMNS, AMOUNT_PRED_COL, SUCCESS_PRED_COL
from data_prep.scaling_data import scale_split_data
from utils.logger import get_logger


logger = get_logger("modelling_pipeline", "logs/modelling_pipeline.log")


def seperate_prediction_data(data: pd.DataFrame, save_data: bool) -> pd.DataFrame:
    """
    Seperate out the data that needs to be predicted per target (for which we don't have values).
    We add the save_data boolean as we want to be able to not save seperated pre_sync_data as this cannot be used later.
    """
    amount_mask = data[AMOUNT_PRED_COL].isna()
    succ_mask = data[SUCCESS_PRED_COL].isna()
    proportion_mask = data[PROPORTION_PRED_COLUMNS].isna().any(axis=1)
    combined_mask = amount_mask | succ_mask | proportion_mask

    amount_pred_data = data[amount_mask]
    succ_pred_data = data[succ_mask]
    proportion_pred_data = data[proportion_mask]

    data = data[~combined_mask]

    if save_data:
        # write pred data to file for later use
        path = "data/needs_predicting/"
        amount_pred_data.to_csv(f"{path}amount_pred_data.csv")
        succ_pred_data.to_csv(f"{path}succ_pred_data.csv")
        proportion_pred_data.to_csv(f"{path}proportion_pred_data.csv")
        logger.info("Successfully wrote entries needing predicting to file")

    logger.info("Successfully seperated out entries needing predicting")
    logger.info(
        f"Number of rows with NaN values remaining: {data.isna().any(axis=1).sum()}"
    )
    return data


def run_models(pre_sync_data: pd.DataFrame):
    features = [
        col
        for col in pre_sync_data.columns
        if col not in (PROPORTION_PRED_COLUMNS + [AMOUNT_PRED_COL, SUCCESS_PRED_COL, "account_id", "mean_account_arpc"])
    ]
    
    # seperate out data that needs predicting
    pre_sync_data = seperate_prediction_data(data=pre_sync_data, save_data=False)

    train_scaled, test_scaled = scale_split_data(
        data=pre_sync_data,
        features=features,
        test_size=0.3,
        random_state=42,
    )
