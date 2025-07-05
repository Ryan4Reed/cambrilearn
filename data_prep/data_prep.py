import pandas as pd
from utils.logger import get_logger
import os

reference_date = pd.Timestamp("2024-07-01")
sync_date = pd.Timestamp("2024-02-01")

###########################################################
#################GENERAL DATA PREP METHODS#################
###########################################################


# def setup_logging():
#     """
#     Setup logger.
#     """
#     os.makedirs("logs", exist_ok=True)

#     # Create a logger specific for this file
#     logger = logging.getLogger("data_prep")
#     logger.setLevel(logging.INFO)

#     # Create file handler
#     file_handler = logging.FileHandler("logs/data_prep.log", mode="w")
#     file_handler.setLevel(logging.INFO)
#     formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
#     file_handler.setFormatter(formatter)
#     if not logger.hasHandlers():
#         logger.addHandler(file_handler)
#     return logger


logger = get_logger("data_prep", "logs/data_prep.log")


def drop_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Drop columns that will not form part of analysis.
    """
    data = data.drop(columns=columns)
    logger.info(f"Successfully dropped the following columns: {columns}")
    return data


def convert_entries_to_nan(
    data: pd.DataFrame, column: str, entries: list[str]
) -> pd.DataFrame:
    """
    Converts entries for a particular column to NaN based on value match.
    """
    affected_count = data[column].isin(entries).sum()
    data[column] = data[column].replace(entries, None)
    logger.info(
        f"Converted {affected_count} entries matching: {entries} to NaN from column: {column}"
    )
    return data


def convert_nan_entries_to_unassigned(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Converts NaN entries for a particular column to Unassigned.
    """
    affected_count = data[column].isna().sum()
    data[column] = data[column].fillna("Unassigned")
    logger.info(
        f"Converted {affected_count} NaN entries to Unassigned from column: {column}"
    )
    return data


def convert_to_datetime(data: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    """
    Converts columns to pandas datetime object
    """
    data[date_columns] = data[date_columns].apply(pd.to_datetime)
    logger.info(f"Converted the following columns to datetime: {date_columns}")
    return data


def drop_rows_outside_period(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all rows for which date_received is on and after cutoff_date.
    """
    init_count = len(data)
    data = data[data["date_received"] < reference_date]
    logger.info(
        f"Dropped {init_count - len(data)} rows with date_received on/after {reference_date}"
    )
    return data


def drop_rows_with_nan(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Drop rows based on NaN entries wrt a particular set of columns.
    """
    init_count = len(data)
    data = data.dropna(subset=columns)
    logger.info(f"Dropped {init_count - len(data)} rows with NaN in columns: {columns}")
    return data


###########################################################
#############CLAIMS_DATA SPECIFIC METHODS##################
###########################################################


def get_pre_sync_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    split according to sync_date.
    """
    pre_data = data[data["date_received"] < sync_date]
    logger.info(f"Successfully split data")
    logger.info(f"Length of complete dataset: {len(data)}")
    logger.info(f"Length of presync dataset: {len(pre_data)}")
    return pre_data


def recreate_months_since_joined(
    data: pd.DataFrame, col_name: str, date: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Recreate the months_since_joined column based on up to date data.
    """
    data = drop_columns(data=data, columns=[col_name])
    data[col_name] = (date.year - data["start_date"].dt.year) * 12 + (
        date.month - data["start_date"].dt.month
    )
    logger.info(f"Successfully created column: {col_name}")
    return data


def recreate_last_year_claim_count(
    data: pd.DataFrame, col_name: str, date: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Recreate last_year_claim_count column based on up to date data.
    """
    data = drop_columns(data=data, columns=[col_name])
    cutoff_date = date - pd.DateOffset(months=12)
    counts = data[data["date_received"] >= cutoff_date].groupby("account_id").size()
    # map counts back to df
    data["last_year_claim_count"] = data["account_id"].map(counts).fillna(0).astype(int)
    logger.info(f"Successfully created column: {col_name}")

    return data


def recreate_months_since_last_claim(
    data: pd.DataFrame, col_name: str, date: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Recreate months_since_last_claim column based on up to date data.
    """
    data = drop_columns(data=data, columns=[col_name])
    latest_claim_date = data.groupby("account_id")["date_received"].max()

    months_diff = (date.year - latest_claim_date.dt.year) * 12 + (
        date.month - latest_claim_date.dt.month
    )
    data["months_since_last_claim"] = data["account_id"].map(months_diff).astype(int)
    logger.info(f"Successfully created column: {col_name}")
    return data


def recreate_dry_months(data: pd.DataFrame, col_names: list[str]) -> pd.DataFrame:
    """
    Recreate dry_months column based on up to date data.
    Also adding four_years_dry column.
    """
    data = drop_columns(data=data, columns=col_names)
    mapping = {
        "two_months_dry": 2,
        "six_months_dry": 6,
        "one_year_dry": 12,
        "two_years_dry": 24,
        "four_years_dry": 48,
    }
    for column in col_names + ["four_years_dry"]:
        data[column] = (data["months_since_last_claim"] > mapping[column]).astype(int)

    logger.info(f"Successfully created columns: {col_names}")
    return data


def recreate_is_closed(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Recreate is_close based on cutoff date.
    """
    data = drop_columns(data=data, columns=[col_name])
    result_date = reference_date - pd.DateOffset(
        months=17
    )  # anything 18 months prior or more would be closed
    data[col_name] = data["date_received"] < result_date
    logger.info(f"Successfully created column: {col_name}")
    return data


def encode_industry_segment_proportions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the industry_segment column by created columns for each category
    containing the proportion of an accounts entries represented by the category.
    """
    # make entries lowercase
    data["industry_segment"] = data["industry_segment"].str.lower()

    # Compute proportions
    proportions = (
        pd.crosstab(data["account_id"], data["industry_segment"])
        .div(data.groupby("account_id").size(), axis=0)
        .add_prefix("industry_segment_")
        .reset_index()
    )

    data = data.merge(proportions, on="account_id", how="left")
    logger.info("Successfully adding industry_segment proportion columns")
    return data


def date_error_proportion_column(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Create column indicating portion of date entries that have issues where either date
    is NaN, or date sequencing is invalid.
    """
    # Add the flag column to data
    data["date_issue"] = (
        (data["start_date"] > data["date_received"])
        | (data["loss_date"] > data["date_received"])
        | (data["start_date"] > data["loss_date"])
        | (data["loss_date"].isna())
    )

    # Compute and assign proportion per account
    data[col_name] = data.groupby("account_id")["date_issue"].transform("mean")
    logger.info(f"Successfully created column: {col_name}")
    return data


def days_between_loss_and_claim(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Calculate the average difference between loss_date and date_received for each account.
    We will exclude entries where loss_date > date_received.
    """
    avg_diff_per_account = (
        data.loc[~data["date_issue"]]  # exclude entries with date issue
        .assign(days_diff=(lambda df: (df["date_received"] - df["loss_date"]).dt.days))
        .groupby("account_id")["days_diff"]
        .mean()
    )

    data[col_name] = data["account_id"].map(avg_diff_per_account).fillna(-1)
    logger.info(f"Successfully created column: {col_name} (NaNs filled with -1)")
    return data


def engineer_claims_features(
    data: pd.DataFrame, date: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    All methods for adding engineer features for claims data
    """
    data = recreate_months_since_joined(
        data=data, col_name="months_since_joined", date=date
    )

    data = recreate_last_year_claim_count(
        data=data, col_name="last_year_claim_count", date=date
    )
    data = recreate_months_since_last_claim(
        data=data, col_name="months_since_last_claim", date=date
    )
    data = recreate_dry_months(
        data=data,
        col_names=["two_months_dry", "six_months_dry", "one_year_dry", "two_years_dry"],
    )

    # Create column indicating portion of date entries that have issues
    data = date_error_proportion_column(data=data, col_name="date_issue_proportion")

    # Create one hot encoded type columns with industry segment proportions
    data = encode_industry_segment_proportions(data=data)
    data = drop_columns(data=data, columns=["industry_segment"])

    # average diff (in days) between loss_date and date_received
    data = days_between_loss_and_claim(
        data=data, col_name="days_between_loss_and_claim"
    )

    # drop date columns now that we no longer need them
    data = drop_columns(
        data=data, columns=["date_received", "loss_date", "start_date", "date_issue"]
    )

    logger.info(f"Successfully engineered claims features")
    return data


def collapse_data(
    data: pd.DataFrame, col_name: str, handle_is_closed: bool = False
) -> pd.DataFrame:
    """
    Collapse data so that we have one row per account
    """
    if handle_is_closed:
        # Find account_ids where any is_closed is true
        closed_accounts = data.query("is_closed")[col_name].unique()

        # Set is_closed to true for those account_ids
        data = data.assign(
            is_closed=data["is_closed"] | data[col_name].isin(closed_accounts)
        )

    data = data.drop_duplicates()

    dupes = data[col_name].value_counts()
    dupes = dupes[dupes > 1].index.tolist()

    if len(dupes) > 1:
        raise ValueError("Failed to collapse data to account level dateset")
    logger.info(f"Successfully collapsed data to {col_name} level dataset")
    return data


def one_hot_encode_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    One-hot encodes the specified columns in the DataFrame.
    Resulting columns are prefixed with the original column name and originals dropped.
    """
    # Make categories lowercase for consistend column naming
    for col in columns:
        data[col] = data[col].str.lower()

    dummies = pd.get_dummies(data[columns], prefix=columns)
    dummies = dummies.astype(int)

    # Drop original columns and join dummies
    data = data.drop(columns=columns).join(dummies)
    logger.info(f"Successfully one-hot encoded columns: {columns}")
    return data


def prep_claims_data() -> pd.DataFrame:
    logger.info("\n==================== CLAIMS DATA ====================")
    data = pd.read_csv("data/raw/all_claims_data.csv")
    ##########################
    ########Clean Data########
    ##########################
    # drop irrelevant columns
    data = drop_columns(
        data=data, columns=["case_number", "is_deleted", "type", "account_name"]
    )

    # Convert unknown account_id entries to NaN and drop all NaN
    data = convert_entries_to_nan(
        data=data, column="account_id", entries=["Unknown Account"]
    )
    data = drop_rows_with_nan(data=data, columns=["account_id"])

    # convert illogical loss_date entries to NaN
    data = convert_entries_to_nan(
        data=data, column="loss_date", entries=["3120-04-03", "3130-04-03"]
    )

    # convert date columns to datetime
    data = convert_to_datetime(
        data=data, date_columns=["date_received", "loss_date", "start_date"]
    )

    # drop rows with date_received outside period of interest
    data = drop_rows_outside_period(data=data)

    # Convert NaN industry_segment entries to Unassigned
    data = convert_nan_entries_to_unassigned(data=data, column="industry_segment")

    # Clean up commercial_subtype
    data = drop_rows_with_nan(data=data, columns=["commercial_subtype"])

    ##########################
    ####Engineer Features#####
    ##########################
    # This can happen prior to split
    data = recreate_is_closed(data=data, col_name="is_closed")

    # get entries with date_received prior to sync date
    pre_sync_data = get_pre_sync_data(data=data)

    # We need to calculate feature wrt to both dataset
    # features from pre_sync_data will be use during traing and from complete data during testing
    data = engineer_claims_features(data, reference_date)
    pre_sync_data = engineer_claims_features(pre_sync_data, sync_date)

    # collapse datasets to one row per account_id
    acc_lev_data = collapse_data(
        data=data, col_name="account_id", handle_is_closed=True
    )
    acc_lev_pre_sync_data = collapse_data(
        data=pre_sync_data, col_name="account_id", handle_is_closed=True
    )

    # one_hot_encode commercial_subtype
    acc_lev_data = one_hot_encode_columns(
        data=acc_lev_data, columns=["commercial_subtype"]
    )
    acc_lev_pre_sync_data = one_hot_encode_columns(
        data=acc_lev_pre_sync_data, columns=["commercial_subtype"]
    )

    logger.info("Successfully prepped claims_data")
    return data, acc_lev_data, acc_lev_pre_sync_data


###########################################################
#############ARPC_DATA SPECIFIC METHODS####################
###########################################################


def seperate_arpc_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Seperate out account and industry level information into seperate datasets.
    """
    industry_arpc = data[
        [
            "industry_segment",
            "mean_industry_arpc",
            "median_industry_arpc",
            "industry_hit_success_rate",
        ]
    ]
    industry_arpc = collapse_data(industry_arpc, col_name="industry_segment")

    account_arpc = data[
        [
            "account_id",
            "median_account_arpc",
            "account_hit_success_rate",
        ]
    ]
    account_arpc = collapse_data(account_arpc, col_name="account_id")

    return industry_arpc, account_arpc

def convert_percent_columns_to_fraction(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Convert percentage columns to fraction.
    """
    for col in columns:
        data[col] = data[col] / 100
    return data

def prep_arpc_data() -> pd.DataFrame:
    logger.info("\n==================== ARPC DATA ====================")
    data = pd.read_csv("data/raw/arpc_values.csv")
    # drop irrelevant columns
    data = drop_columns(
        data=data, columns=["account_name", "commercial_subtype", "mean_account_arpc"]
    )
    # change percent to fraction
    data = convert_percent_columns_to_fraction(data=data, columns=['account_hit_success_rate', 'industry_hit_success_rate'])
    # seperate out account and industry level information
    industry_arpc, account_arpc = seperate_arpc_data(data=data)
    logger.info("Successfully prepped arpc_data")
    return industry_arpc, account_arpc


###########################################################
############ACCOUNT_REV_DIST SPECIFIC METHODS##############
###########################################################


def drop_invalid_proportions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drops all rows for account_ids that have any revenue_proportion outside [0, 1].
    """
    bad_accounts = data.loc[
        (data["revenue_proportion"] > 1) | (data["revenue_proportion"] < 0),
        "account_id",
    ].unique()

    count_before = len(data["account_id"].unique())
    # Drop all rows for those accounts
    data = data[~data["account_id"].isin(bad_accounts)].copy()

    logger.info(
        f"Dropped {count_before - len(data['account_id'].unique())} accounts with invalid proportions."
    )
    return data


def prep_account_revenue_dist_data() -> pd.DataFrame:
    logger.info("\n==================== ACCOUNT_REVENUE_DIST DATA ====================")
    data = pd.read_csv("data/raw/account_revenue_distributions.csv")
    # drop irrelevant columns
    data = drop_columns(data=data, columns=["account_name", "claim_count"])

    # drop accounts where proportions per month are >1 < 0
    data = drop_invalid_proportions(data=data)
    # drop accounts where proportions are nan
    data = drop_rows_with_nan(data=data, columns=["revenue_proportion"])

    logger.info("Successfully prepped account_revenue_dist_data")
    return data


###########################################################
############INDUSTRY_REV_DIST SPECIFIC METHODS#############
###########################################################


def prep_industry_revenue_dist_data() -> pd.DataFrame:
    logger.info(
        "\n==================== INDUSTRY_REVENUE_DIST DATA ===================="
    )
    data = pd.read_csv("data/raw/industry_revenue_distributions.csv")
    # drop irrelevant columns
    data = drop_columns(data=data, columns=["claim_count"])
    logger.info("Successfully prepped industry_revenue_dist_data")
    return data


###########################################################
############Combining All Datasets#############
###########################################################

col_name_mapping = {
    "industry_segment_commercial": "Commercial",
    "industry_segment_dealership": "Dealership",
    # "industry_segment_insurance company": "Insurance Company",
    "industry_segment_municipality": "Municipality",
    "industry_segment_rental": "Rental",
}


def add_weighted_industry_arpc_metrics(
    claims_data: pd.DataFrame, industry_arpc: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds weighted industry-level arpc metrics to claims_data based on industry proportions.
    Excludes 'industry_segment_unassigned', and 'industry_segment_car insurance'.
    """
    # Make sure industry_arpc is indexed on industry_segment
    industry_info = industry_arpc.set_index("industry_segment")

    # Pull the proportion values matrix from claims_data
    proportions = claims_data[list(col_name_mapping.keys())].values

    # For each industry-level metric, compute weighted sum
    for metric in [
        "mean_industry_arpc",
        "median_industry_arpc",
        "industry_hit_success_rate",
    ]:
        industry_values = industry_info.loc[
            list(col_name_mapping.values()), metric
        ].values
        weighted_sum = (proportions * industry_values).sum(axis=1)
        claims_data[f"weighted_{metric}"] = weighted_sum

    logger.info("Successfully added weighted industry metrics.")
    return claims_data


def add_account_arpc_metrics(
    data: pd.DataFrame, account_arpc: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds account-level metrics to data based on account_id.
    """
    data = pd.merge(data, account_arpc, on="account_id", how="left")

    return data


def add_ind_rev_distributions(
    data: pd.DataFrame, industry_dist: pd.DataFrame
) -> pd.DataFrame:
    """
    Add 18 columns containing the industry revenue distribution for each month wrt each account_id
    weighted by the proportion to which each industry represents each account.
    """
    # Pivot industry_dist
    industry_month_matrix = (
        industry_dist.pivot(
            index="industry", columns="month", values="revenue_proportion"
        ).reindex(
            index=list(col_name_mapping.values())
        )  # Ensures order matches our mapping
    )

    # Convert to numpy matrix for fast multiplication
    industry_month_values = industry_month_matrix.values

    # Extract account industry proportions
    account_proportions = data[list(col_name_mapping.keys())].values

    # Multiply each account proportions × industry_month_values
    predicted_distribution = account_proportions @ industry_month_values

    # Add these 18 columns to claims_data
    for month in range(1, 19):
        col_name = f"industry_prop_month_{month}"
        data[col_name] = predicted_distribution[:, month - 1]

    logger.info("Successfully added industry revenue distribution columns.")
    return data


def add_acc_rev_distributions(
    data: pd.DataFrame, account_dist: pd.DataFrame
) -> pd.DataFrame:
    """
    Add 18 columns containing the industry revenue distribution for each month wrt each account_id
    """
    # Pivot account_dist
    account_pivot = account_dist.pivot(
        index="account_id", columns="month", values="revenue_proportion"
    )
    account_pivot.columns = [
        f"account_prop_month_{int(month)}" for month in account_pivot.columns
    ]
    # make account_id column again
    account_pivot = account_pivot.reset_index()
    data = data.merge(account_pivot, on="account_id", how="left")

    logger.info("Successfully added account_revenue_distribution columns.")

    return data


def combine_data(
    claims_data: pd.DataFrame,
    industry_arpc: pd.DataFrame,
    account_arpc: pd.DataFrame,
    account_dist: pd.DataFrame,
    industry_dist: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine datasets into single dataset.
    """

    data = add_weighted_industry_arpc_metrics(
        claims_data=claims_data, industry_arpc=industry_arpc
    )
    data = add_account_arpc_metrics(data=data, account_arpc=account_arpc)
    data = add_ind_rev_distributions(data=data, industry_dist=industry_dist)
    data = add_acc_rev_distributions(data=data, account_dist=account_dist)

    return data


def drop_median_arpc_outliers(data: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Drop outliers based on value of median_account_arpc.
    """

    data = data[data["median_account_arpc"] <= threshold]
    logger.info("Successfully removed outliers.")
    return data


def run_data_prep():
    try:
        print(">Starting data preparation pipeline")
        claim_level_data, acc_level_data, acc_level_pre_sync_data = (
            prep_claims_data()
        )

        industry_arpc, account_arpc = prep_arpc_data()
        account_dist = prep_account_revenue_dist_data()
        industry_dist = prep_industry_revenue_dist_data()

        acc_level_data = combine_data(
            claims_data=acc_level_data,
            industry_arpc=industry_arpc,
            account_arpc=account_arpc,
            account_dist=account_dist,
            industry_dist=industry_dist,
        )
        acc_level_pre_sync_data = combine_data(
            claims_data=acc_level_pre_sync_data,
            industry_arpc=industry_arpc,
            account_arpc=account_arpc,
            account_dist=account_dist,
            industry_dist=industry_dist,
        )

        acc_level_pre_sync_data = drop_median_arpc_outliers(data=acc_level_pre_sync_data, threshold=5000)

        # Write both DataFrames to csv
        acc_level_data.to_csv("data/prepped/acc_level_data.csv", index=False)
        acc_level_pre_sync_data.to_csv("data/prepped/acc_level_pre_sync_data.csv", index=False)
        claim_level_data.to_csv("data/prepped/claim_level_data.csv", index=False)

        logger.info("Successfully wrote prepped data files to data/prepped folder.")
        print(">Data preparation pipeline executed successfully")
        return acc_level_data, acc_level_pre_sync_data, claim_level_data

    except Exception as error:
        logger.error(error)
        raise error


def load_cached_data() -> pd.DataFrame:
    """
    Loading saved data from file.
    """
    print('>Loading data from file')
    acc_level_data = pd.read_csv("data/prepped/acc_level_data.csv")
    acc_level_pre_sync_data = pd.read_csv("data/prepped/acc_level_pre_sync_data.csv")
    claim_level_data = pd.read_csv("data/prepped/claim_level_data.csv")
    print('>Data loaded successfully')
    return acc_level_data, acc_level_pre_sync_data, claim_level_data