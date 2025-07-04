from data_prep.data_prep import run_data_prep
from data_prep.feature_validation import generate_pearson_correlations
from models.modelling_pipeline import run_models
from utils.logger import get_logger
import traceback

if __name__ == "__main__":
    try:
        print(">Starting Application")
        logger = get_logger('main', 'logs/main.log')
        data, pre_sync_data = run_data_prep()
        generate_pearson_correlations(pre_sync_data)
        run_models(pre_sync_data=pre_sync_data)
        print(">Application Successful")

    except Exception as error:
        logger.info(f"Error: {error}\n{traceback.format_exc()}")
        raise 
