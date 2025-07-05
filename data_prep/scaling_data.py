import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from utils.logger import get_logger

logger = get_logger("scaling_data", "logs/scaling_data.log")


def scale_split_data(
    data: pd.DataFrame, features: list[str], test_size: float, random_state: int
):
    """
    Scale data and split in train and test sets.
    """
    logger.info(f"\n================== Scaling Data ==================")

    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train rows: {len(train_data)}, Test rows: {len(test_data)}")

    # Fit scaler on training features
    scaler = StandardScaler()
    scaler.fit(train_data[features])

    # Transform train and test features
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()
    train_scaled[features] = scaler.transform(train_data[features])
    test_scaled[features] = scaler.transform(test_data[features])
    logger.info("Applied scaling to train and test sets.")

    path = "data/prepped/scaled_splits/"
    train_scaled.to_csv(f"{path}train.csv", index=False)
    test_scaled.to_csv(f"{path}test.csv", index=False)

    # Save scaler
    joblib.dump(scaler, "data/scaler/standard_scaler.pkl")

    logger.info(f"Saved scaled and split data and scaler to file")
    return train_scaled, test_scaled
