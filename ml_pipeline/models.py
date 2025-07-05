import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning

from utils.logger import get_logger

warnings.filterwarnings("ignore", category=ConvergenceWarning)
logger = get_logger("models", "logs/models.log")


####################################################################
##########################ELASTICNET################################
####################################################################


def plot_gridsearch_heatmap(
    results,
    param_x,
    param_y,
    score="mean_test_score",
    title="",
    save_path="heatmap.png",
):
    """
    Plot heatmap of mean_test_score over two hyperparameter grid dimensions.
    Handles duplicates by averaging over other parameters.
    """
    if param_x not in results.columns or param_y not in results.columns:
        raise ValueError(f"Columns {param_x} or {param_y} not in results DataFrame.")

    # Group by param_x and param_y, averaging over other hyperparameters
    pivot_df = results.groupby([param_y, param_x])[score].mean().reset_index()

    # Convert negative RMSE back to positive
    pivot_df[score] = -pivot_df[score]

    pivot = pivot_df.pivot(index=param_y, columns=param_x, values=score)
    # round labels
    pivot.index = pivot.index.round(2)
    pivot.columns = pivot.columns.round(2)

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt="0.0f", cmap="viridis")
    plt.title(title or f"Grid Search Heatmap: {score}")
    plt.xlabel(param_x.replace("param_", ""))
    plt.ylabel(param_y.replace("param_", ""))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Saved heatmap plot to: {save_path}")


def train_and_evaluate_elasticnet(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features: list[str],
    target: str,
    random_state: int,
    n_splits: int = 5,
):
    """
    Train and evaluate ElasticNet with GridSearchCV and KFold.
    """
    logger.info(
        f"\n================== ElasticNet Training for Target: {target} =================="
    )

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Define ElasticNet and parameter grid
    model = ElasticNet(max_iter=10000)
    param_grid = {"alpha": np.logspace(-3, 1, 20), "l1_ratio": np.linspace(0.1, 0.9, 9)}
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
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
    m_path = f"ml_pipeline/elasticnet/best_model/{target}.pkl"
    joblib.dump(best_model, m_path)
    logger.info(f"Saved trained model to: {m_path}")

    # Save grid search heatmap
    heatmap_path = f"ml_pipeline/elasticnet/gs_results/{target}_heatmap.png"
    plot_gridsearch_heatmap(
        results=pd.DataFrame(grid_search.cv_results_),
        param_x="param_alpha",
        param_y="param_l1_ratio",
        score="mean_test_score",
        title=f"ElasticNet Grid Search for {target}",
        save_path=heatmap_path,
    )

    logger.info("================== DONE ==================\n")

    return best_model, rmse, r2


####################################################################
###########################XGBOOST##################################
####################################################################


def plot_score_vs_param(results, param, target, save_path):
    """
    Plot mean_test_score (converted to RMSE) vs single hyperparameter.
    """
    results = results[[param, "mean_test_score"]].copy()
    results["mean_test_score"] = -results["mean_test_score"]
    results = results.groupby(param).mean().reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(results[param], results["mean_test_score"], marker="o")
    plt.title(f"{target}: Mean RMSE vs {param}")
    plt.xlabel(param)
    plt.ylabel("Mean RMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Saved score vs {param} plot to: {save_path}")


def train_and_evaluate_xgboost(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features: list[str],
    target: str,
    random_state: int,
    n_splits: int = 5,
):
    """
    Train and evaluate XGBoost Regressor with GridSearchCV and KFold.
    """
    logger.info(
        f"\n================== XGBoost Training for Target: {target} =================="
    )

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Define XGBoost model and parameter grid
    model = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=random_state, verbosity=0
    )

    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [50, 100, 200],
        "subsample": [0.7, 0.9, 1.0],
    }
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Log best params
    logger.info(f"Best Parameters: {grid_search.best_params_}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Test R^2: {r2:.4f}")

    # Save model
    m_path = f"ml_pipeline/xgboost/best_model/{target}.pkl"
    joblib.dump(best_model, m_path)
    logger.info(f"Saved trained model to: {m_path}")

    # Save line plots for each hyperparameter individually
    results = pd.DataFrame(grid_search.cv_results_)
    for param in param_grid.keys():
        param_col = f"param_{param}"
        if param_col in results.columns:
            plot_score_vs_param(
                results=results,
                param=param_col,
                target=target,
                save_path=f"ml_pipeline/xgboost/gs_results/{target}_vs_{param}.png",
            )

    logger.info("================== DONE ==================\n")

    return best_model, rmse, r2


def clr_transform(Y):
    """
    Centered log-ratio transformation.
    """
    Y_safe = np.clip(Y, 1e-10, None)
    log_Y = np.log(Y_safe)
    gm = log_Y.mean(axis=1, keepdims=True)
    return log_Y - gm


def clr_inverse(Y_clr):
    """
    Inverse of CLR transformation to get proportions summing to 1.
    """
    exp = np.exp(Y_clr)
    return exp / exp.sum(axis=1, keepdims=True)


def train_and_evaluate_xgboost_clr(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features: list[str],
    target_columns: list[str],
    random_state: int,
    n_splits: int = 5,
):
    """
    Xgboost regression for vector target. Making use of centered log-ratio transformations.
    """

    logger.info(
        f"\n================== XGBoost CLR Regression for Targets: {target_columns} =================="
    )

    # Extract features and targets
    X_train = train_data[features]
    Y_train_raw = train_data[target_columns].values
    X_test = test_data[features]
    Y_test_raw = test_data[target_columns].values
    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # CLR-transform targets
    Y_train_clr = clr_transform(Y_train_raw)

    # Define base XGBoost model
    base_model = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=random_state, verbosity=0
    )

    # Hyperparameter grid
    param_grid = {
        "estimator__max_depth": [3, 4, 5],
        "estimator__learning_rate": [0.01, 0.05, 0.1],
        "estimator__n_estimators": [50, 100, 200],
        "estimator__subsample": [0.7, 0.9, 1.0],
    }

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Multitarget regression via GridSearchCV with wrapper
    from sklearn.multioutput import MultiOutputRegressor

    model = MultiOutputRegressor(base_model)

    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1
    )

    logger.info("Starting grid search...")
    grid_search.fit(X_train, Y_train_clr)

    best_model = grid_search.best_estimator_

    logger.info(f"Best Parameters: {grid_search.best_params_}")

    # Predict in CLR space
    Y_pred_clr = best_model.predict(X_test)
    # Inverse CLR to get valid proportions
    Y_pred_proportions = clr_inverse(Y_pred_clr)

    # Evaluate RMSE over all components
    rmse = np.sqrt(mean_squared_error(Y_test_raw, Y_pred_proportions))
    r2 = r2_score(Y_test_raw, Y_pred_proportions)

    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Test R^2: {r2:.4f}")

    # Save model
    m_path = f"ml_pipeline/xgboost/best_model/proportion_rev.pkl"
    joblib.dump(best_model, m_path)
    logger.info(f"Saved trained model to: {m_path}")

    logger.info("================== DONE ==================\n")

    return best_model, rmse, r2
