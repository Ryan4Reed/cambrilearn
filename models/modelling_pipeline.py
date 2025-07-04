import pandas as pd

from utils.utils import PROPORTION_PRED_COLUMNS, AMOUNT_PRED_COL, SUCCESS_PRED_COL
from data_prep.scaling_data import scale_split_data
from utils.logger import get_logger
from models.models import train_and_evaluate_elasticnet, train_and_evaluate_xgboost, train_and_evaluate_xgboost_clr


logger = get_logger("modelling_pipeline", "logs/modelling_pipeline.log")
random_state = 42

def get_features(data: pd.DataFrame) -> list:
    """
    Dynamic method to extract features columns
    """
    features = [
        col
        for col in data.columns
        if col not in (PROPORTION_PRED_COLUMNS + [AMOUNT_PRED_COL, SUCCESS_PRED_COL, "account_id"])
    ]
    return features

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
    # seperate out data that needs predicting
    pre_sync_data = seperate_prediction_data(data=pre_sync_data, save_data=False)
    # drop account_id
    pre_sync_data = pre_sync_data.drop(columns=['account_id'])
    # check all columns numberic
    numeric_check = pre_sync_data.apply(pd.api.types.is_numeric_dtype)
    if not numeric_check.all():
        non_numeric_columns = pre_sync_data.columns[~numeric_check].tolist()
        error = f'There are non-numeric columns in Dataset:{non_numeric_columns}'
        logger.error(error)
        raise ValueError(error)


    train_scaled, test_scaled = scale_split_data(
        data=pre_sync_data,
        features=get_features(pre_sync_data),
        test_size=0.3,
        random_state=random_state,
    )

    run_ammount_models(train_data=train_scaled, test_data=test_scaled)
    run_success_models(train_data=train_scaled, test_data=test_scaled)
    run_proportion_models(train_data=train_scaled, test_data=test_scaled)


def run_ammount_models(train_data: pd.DataFrame, test_data: pd.DataFrame):
    print('>Running amount models')
    features = get_features(train_data)

    ##########################################

    print("----->Training elasticnet_model")
    elasticnet_model, elasticnet_rmse, elasticnet_r2 = train_and_evaluate_elasticnet(
        train_data=train_data,
        test_data=test_data,
        features=features,
        target=AMOUNT_PRED_COL,
        random_state=random_state

    )
    print("----->Training xgboost_model")
    xgboost_model, xgboost_rmse, xgboost_r2 = train_and_evaluate_xgboost(
        train_data=train_data,
        test_data=test_data,
        features=features,
        target=AMOUNT_PRED_COL,
        random_state=random_state

    )
    print('>Amount models ran successfully')

def run_success_models(train_data: pd.DataFrame, test_data: pd.DataFrame):
    print('>Running success models')
    features = get_features(train_data)

    ##########################################

    print("----->Training elasticnet_model")
    elasticnet_model, elasticnet_rmse, elasticnet_r2 = train_and_evaluate_elasticnet(
        train_data=train_data,
        test_data=test_data,
        features=features,
        target=SUCCESS_PRED_COL,
        random_state=random_state

    )
    print("----->Training xgboost_model")
    xgboost_model, xgboost_rmse, xgboost_r2 = train_and_evaluate_xgboost(
        train_data=train_data,
        test_data=test_data,
        features=features,
        target=SUCCESS_PRED_COL,
        random_state=random_state

    )
    print('>Amount models ran successfully')


def run_proportion_models(train_data: pd.DataFrame, test_data: pd.DataFrame):
    print('>Running proportion model')
    features = get_features(train_data)

    ##########################################

    print("----->Training xgboost_CLR_model")
    xgboost_model, xgboost_rmse, xgboost_r2 = train_and_evaluate_xgboost_clr(
        train_data=train_data,
        test_data=test_data,
        features=features,
        target_columns=PROPORTION_PRED_COLUMNS,
        random_state=random_state

    )
    print('>Proportion model ran successfully')
