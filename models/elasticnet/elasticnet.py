import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib as plt
import seaborn as sns

from utils.logger import get_logger
from utils.utils import PROPORTION_PRED_COLUMNS, AMOUNT_PRED_COL, SUCCESS_PRED_COL

logger = get_logger('models', 'logs/models.log')


def train_and_evaluate_elasticnet(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    test_size: float = 0.3,
    random_state: int = 42, # for fair comparison between models
    n_splits: int = 5,
):
    """
    Train and evaluate ElasticNet with GridSearchCV and KFold.
    """
    logger.info(f"\n================== ElasticNet Training for Target: {target} ==================")

    # Drop rows with NaN target
    data = data.dropna(subset=[target])
    logger.info(f"Using {len(data)} rows after dropping NaNs in target.")

    X = data[features]
    y = data[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Define ElasticNet and parameter grid
    model = ElasticNet(max_iter=10000)
    param_grid = {
        "alpha": np.logspace(-3, 1, 20),
        "l1_ratio": np.linspace(0.1, 0.9, 9)
    }
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    logger.info(f"Best alpha: {grid_search.best_params_['alpha']:.4f}")
    logger.info(f"Best l1_ratio: {grid_search.best_params_['l1_ratio']:.4f}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Test R^2: {r2:.4f}")

    # Save model
    m_path = f'models/elasticnet/best_model/{target}.pkl'
    joblib.dump(best_model, m_path)
    logger.info(f"Saved trained model to: {m_path}")

    # Save grid search results
    results_df = pd.DataFrame(grid_search.cv_results_)
    gs_path = f'models/elasticnet/gs_results/{target}.csv'
    results_df.to_csv(gs_path, index=False)
    logger.info(f"Saved GridSearchCV results to: {gs_path}")


    logger.info("================== DONE ==================\n")

    return best_model, rmse, r2


def plot_gridsearch_heatmap(results_df, param_x, param_y, score='mean_test_score', title="", save_path="heatmap.png"):
    """
    Plot heatmap of mean_test_score over two hyperparameter grid dimensions.
    """
    if param_x not in results_df.columns or param_y not in results_df.columns:
        raise ValueError(f"Columns {param_x} or {param_y} not in results DataFrame.")

    pivot = results_df.pivot(
        index=param_y,
        columns=param_x,
        values=score
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title(title or f"Grid Search Heatmap: {score}")
    plt.xlabel(param_x.replace('param_', ''))
    plt.ylabel(param_y.replace('param_', ''))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Saved heatmap plot to: {save_path}")



def run_elasticnet():
    data = pd.read_csv("data/prepped/final_pre_sync_data.csv")

    # All target columns across all tasks
    all_targets = [
        AMOUNT_PRED_COL,
        SUCCESS_PRED_COL,
    ] + PROPORTION_PRED_COLUMNS

    features = [
        col
        for col in data.columns
        if col
        not in (
            all_targets
            + [
                "account_id",
                "mean_account_arpc",
            ]
        )
    ]

    ##########################################

    # median_account_arpc
    median_model, median_rmse, median_r2, median_results = train_and_evaluate_elasticnet(
        data=data,
        features=features,
        target=AMOUNT_PRED_COL,

    )

    # # account_hit_success_rate
    hit_model, hit_rmse, hit_r2, hit_results = train_and_evaluate_elasticnet(
        data=data,
        features=features,
        target=SUCCESS_PRED_COL,
    )