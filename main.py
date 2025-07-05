import traceback
import argparse
from data_prep.data_prep import run_data_prep, load_cached_data
from data_prep.feature_validation import generate_pearson_correlations
from ml_pipeline.modelling_pipeline import run_models
from ml_pipeline.predictions import make_predictions
from revenue.revenue import determine_revenue
from utils.logger import get_logger


logger = get_logger("main", "logs/main.log")

def parse_args():
    """
    Define flags
    """
    parser = argparse.ArgumentParser(
        description="Run your pipeline with optional steps."
    )

    parser.add_argument(
        "--run-data-prep",
        action="store_true",
        help="Include data preparation step",
    )
    parser.add_argument(
        "--run-models",
        action="store_true",
        help="Include modeling step",
    )
    parser.add_argument(
        "--get-revenue",
        action="store_true",
        help="Include prediction step",
    )

    return parser.parse_args()


def main():
    """
    Main orchestration method for ML pipeline
    """
    try:
        print(">Starting Application")
        args = parse_args()
        if args.run_data_prep:
            acc_level_data, acc_level_pre_sync_data, claim_level_data = run_data_prep()
            generate_pearson_correlations(acc_level_pre_sync_data)
        else:
            acc_level_data, acc_level_pre_sync_data, claim_level_data = load_cached_data()

        if args.run_models:
            run_models(pre_sync_data=acc_level_pre_sync_data)

        if args.get_revenue:
            acc_level_data = make_predictions(acc_level_data=acc_level_data)
            determine_revenue(acc_level_data=acc_level_data, claim_level_data=claim_level_data)
        print(">Application Successful")

    except Exception as error:
        logger.info(f"Error: {error}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
