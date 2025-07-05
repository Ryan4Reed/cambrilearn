import pandas as pd
import joblib

from utils.utils import PROPORTION_PRED_COLUMNS, AMOUNT_PRED_COL, SUCCESS_PRED_COL
from utils.logger import get_logger
from ml_pipeline.modelling_pipeline import get_features

logger = get_logger("predictions", "logs/predictions.log")


def load_pkl(path: str):
    """
    Load pkl file.
    """
    pkl = joblib.load(path)
    logger.info(f"Successfully loaded pkl: {path}")
    return pkl


def make_amount_predictions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Make amount predictions for data effecting expected revenue using the best performing xgboost model.
    """
    logger.info(f"\n================== Make Amount Predictions ==================")
    # load scaler and model
    scaler = load_pkl("data/scaler/standard_scaler.pkl")
    model = load_pkl("ml_pipeline/xgboost/best_model/median_account_arpc.pkl")
    mask = data[AMOUNT_PRED_COL].isna()
    to_predict = data[mask].drop(columns=["is_closed"])
    logger.info(f"Number of entries to predict: {len(to_predict)}")

    # seperate features from targets
    feature_columns = get_features(to_predict)
    features = to_predict[feature_columns]

    # Scale feature data
    features_scaled = scaler.transform(features)

    # Make predictions
    predictions = model.predict(features_scaled)
    # Assign predictions back into original DataFrame
    data.loc[mask, AMOUNT_PRED_COL] = predictions

    if len(data[data[AMOUNT_PRED_COL].isna()]) > 0:
        error = f"There are NaN values remaining post prediction in {AMOUNT_PRED_COL}"
        logger.error(error)
        raise ValueError(error)

    logger.info("================== Amount Predictions Complete ==================\n")
    return data


def make_nan_mean(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Determine column mean and set nan values to it.
    """
    logger.info(f"Number of values to predict: {len(data[data[column_name].isna()])}")
    mean = data[column_name].mean()
    logger.info(f"Mean for column {column_name}: {mean}")
    data[column_name] = data[column_name].fillna(mean)
    return data, mean


def make_success_predictions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Make success rate predictions for data effecting expected revenue using mean.
    """
    logger.info(
        f"\n================== Make Success Rate Predictions =================="
    )

    data, mean = make_nan_mean(data=data, column_name=SUCCESS_PRED_COL)

    if len(data[data[SUCCESS_PRED_COL].isna()]) > 0:
        error = f"There are NaN values remaining post prediction in {AMOUNT_PRED_COL}"
        logger.error(error)
        raise ValueError(error)

    logger.info(
        "================== Success Rate Predictions Complete ==================\n"
    )
    return data


def make_proportion_predictions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Make revenue proportion predictions for data effecting expected revenue using mean.
    """
    logger.info(
        f"\n================== Make Revenue Proportion Predictions =================="
    )
    means = []
    for colum_name in PROPORTION_PRED_COLUMNS:
        data, mean = make_nan_mean(data=data, column_name=colum_name)
        means.append(mean)
        if len(data[data[colum_name].isna()]) > 0:
            error = (
                f"There are NaN values remaining post prediction in {AMOUNT_PRED_COL}"
            )
            logger.error(error)
            raise ValueError(error)

    logger.info(f"Sum of means for each of the 18 months is equal to: {sum(means)}")
    logger.info(
        "================== Revenue Proportion Predictions Complete ==================\n"
    )
    return data


def make_predictions(acc_level_data: pd.DataFrame):
    """
    Make predictions for entries that effect the expected revenue post 2024-06-30.
    Based on model performance, we use:
    - the xgboost model for predicting median_account_arpc
    - mean across account for account_hit_success_rate
    - per month mean for account_revenue_proportion
    """
    print(">Running prediction pipeline")

    acc_level_data = make_amount_predictions(data=acc_level_data)
    acc_level_data = make_success_predictions(data=acc_level_data)
    acc_level_data = make_proportion_predictions(data=acc_level_data)
    print(">Prediction pipeline ran successfully")
    return acc_level_data
