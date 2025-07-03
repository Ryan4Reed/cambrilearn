import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


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


if __name__ == "__main__":
    data = pd.read_csv("data/prepped/final_pre_sync_data.csv")

    account_month_cols = [f"account_rev_month_{i}" for i in range(1, 19)]

    # All target columns across all tasks
    all_targets = [
        "median_account_arpc",
        "account_hit_success_rate",
    ] + account_month_cols
    features = [col for col in data.columns if col not in all_targets]

    ##########################################
    
    median_acc_arpc_features = [
        col
        for col in data.columns
        if col
        not in (
            all_targets
            + [
                "account_id",
                "mean_account_arpc",
                "median_account_arpc",
                "account_hit_success_rate",
            ]
        )
    ]
    corr_median_arpc = pearson_corr_scalar(
        data, median_acc_arpc_features, "median_account_arpc"
    )
    plot_scalar_corr(corr_median_arpc, title="median_account_arpc")

    ###########################################

    account_success_rate_features = [
        col
        for col in data.columns
        if col
        not in (
            all_targets
            + [
                "account_id",
                "mean_account_arpc",
                "median_account_arpc",
                "account_hit_success_rate",
            ]
        )
    ]
    corr_median_arpc = pearson_corr_scalar(
        data, median_acc_arpc_features, "account_hit_success_rate"
    )
    plot_scalar_corr(corr_median_arpc, title="account_hit_success_rate")

    ############################################

    account_rev_month_features = [
        col
        for col in data.columns
        if col
        not in (
            all_targets
            + [
                "account_id",
                "mean_account_arpc",
                "median_account_arpc",
                "account_hit_success_rate",
            ]
        )
    ]
    rev_month_cols = [f"account_rev_month_{i}" for i in range(1, 19)]
    corr_rev_month = pearson_corr_vector(data, account_rev_month_features, rev_month_cols)
    plot_vector_corr_heatmap(corr_rev_month, "account_rev_month")
    


