from data_prep.data_prep import run_data_prep
from data_prep.feature_validation import generate_pearson_correlations
from models.elasticnet.elasticnet import run_elasticnet
from models.modelling_pipeline import run_models


if __name__ == "__main__":
    data, pre_sync_data = run_data_prep()
    generate_pearson_correlations(pre_sync_data)
    run_models(pre_sync_data=pre_sync_data)
