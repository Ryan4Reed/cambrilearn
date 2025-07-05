import os

PROPORTION_PRED_COLUMNS = [f"account_prop_month_{i}" for i in range(1, 19)]
AMOUNT_PRED_COL = 'median_account_arpc'
SUCCESS_PRED_COL = 'account_hit_success_rate'



def make_folders():
    """
    Instantiate required folder structure.
    """

    folders_to_create = [
        "data_prep",
        "revenue",
        "utils",
        "logs",
        "app",
        "data",
        "data/plots",
        "data/prepped",
        "data/raw",
        "data/revenue",
        "data/scaler",
        "ml_pipeline",
        "ml_pipeline/elasticnet",
        "ml_pipeline/elasticnet/best_model",
        "ml_pipeline/elasticnet/gs_results",
        "ml_pipeline/xgboost",
        "ml_pipeline/xgboost/best_model",
        "ml_pipeline/xgboost/gs_results",
    ]

    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)
        print(f"Ensured folder exists: {folder}")