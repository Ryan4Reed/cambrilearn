import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from utils.utils import PROPORTION_PRED_COLUMNS, AMOUNT_PRED_COL, SUCCESS_PRED_COL


def pearson_corr_scalar(
    data: pd.DataFrame, features: list[str], target: str
) -> pd.Series:
    """
    Compute Pearson correlation between each feature for a scalar target,
    excluding rows with NaN target.
    """
    valid_data = data.dropna(subset=[target])
    corr = valid_data[features + [target]].corr()[target].drop(target)
    return corr


def pearson_corr_vector(
    data: pd.DataFrame, features: list[str], target_cols: list[str]
) -> pd.DataFrame:
    """
    Compute Pearson correlation between each feature and each  proportion value.
    Excludes rows with NaN in target.
    """
    results = {}
    for target in target_cols:
        valid_data = data.dropna(subset=[target])
        corr = valid_data[features + [target]].corr()[target].drop(target)
        results[target] = corr
    return pd.DataFrame(results)


def plot_scalar_corr(
    corr_series: pd.Series, title: str = "Pearson Correlation with Target"
):
    """
    Create a horizontal bar plot of feature correlations with a scalar target.
    """
    plt.figure(figsize=(8, max(4, len(corr_series) // 2)))
    corr_series.sort_values().plot(kind="barh")
    plt.title(title)
    plt.xlabel("Pearson Correlation")
    plt.tight_layout()
    filepath = os.path.join("data/plots", title)
    plt.savefig(filepath)
    plt.close()


def plot_vector_corr_heatmap(
    corr_df: pd.DataFrame, title: str = "Feature vs Target Correlations"
):
    """
    Create a heatmap of correlations between features and multiple target columns.
    """
    plt.figure(figsize=(12, max(6, len(corr_df) // 2)))
    sns.heatmap(corr_df, cmap="coolwarm", center=0, annot=False)
    plt.title(title)
    plt.xlabel("Target Columns")
    plt.ylabel("Features")
    plt.tight_layout()
    filepath = os.path.join("data/plots", title)
    plt.savefig(filepath)
    plt.close()


def generate_pearson_correlations(data: pd.DataFrame):
    """
    Generate pearson correlation plots for each target.
    """
    print('>Generating pearson coefficients')
    # All target columns across all tasks
    all_targets = [
        AMOUNT_PRED_COL,
        SUCCESS_PRED_COL,
    ] + PROPORTION_PRED_COLUMNS

    features = [
        col
        for col in data.columns
        if col not in (all_targets + ["account_id"])
    ]

    ##########################################

    corr_median_arpc = pearson_corr_scalar(data, features, AMOUNT_PRED_COL)
    plot_scalar_corr(corr_median_arpc, title=AMOUNT_PRED_COL)

    ###########################################

    corr_median_arpc = pearson_corr_scalar(data, features, SUCCESS_PRED_COL)
    plot_scalar_corr(corr_median_arpc, title=SUCCESS_PRED_COL)

    ############################################

    corr_rev_month = pearson_corr_vector(data, features, PROPORTION_PRED_COLUMNS)
    plot_vector_corr_heatmap(corr_rev_month, "account_prop_month")
    print('>Pearson coefficients generated successfully')
