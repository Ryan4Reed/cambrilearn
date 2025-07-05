# Cambrilearn project

This project is a modular, command-line-driven machine learning pipeline for preparing data, validating features, training models, making predictions, and estimating revenue.  

It is designed to support optional execution of pipeline stages via command-line flags for maximum flexibility in development and production.

---

## Features

#### Ml pipeline:
- **Data Preparation**: Clean and preprocess raw data for modeling
- **Feature Validation**: Generate correlation analyses to validate inputs
- **Model Training**: Train multiple machine learning models for prediction tasks
- **Prediction**: Generate predictions from trained models
- **Revenue Calculation**: Estimate revenue based on prediction outputs
- **Modular Execution**: Run the pipeline end-to-end or individual steps as needed

#### User interface
- See the Frontend Application section below.
---

--

## Project Structure 
```bash
Cambrilearn/
├── data_prep/
│   ├── data_prep.py
│   └── feature_validation.py
│   └── scaling_data.py
├── ml_pipeline/
│   ├── models.py
│   ├── modelling_pipeline.py
│   └── predictions.py
├── revenue/
│   └── revenue.py
├── utils/
│   ├── logger.py
│   └── utils.py
├── logs/
│   └── ...
├── data/
│   └── ...
├── main.py
├── requirements.txt
├── app/
│   ├── app.py

└──
```


## Installation

1. Clone the repository:
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```
3. Install required packages:
```bash
pip install -r requirements.txt
```

---

## Usage of ML Pipeline

#### Preprocessing step
You will need to take the raw datasets namely:
- `all_claims_data.csv`
- `arpc_values.csv`
- `account_revenue_distributions.csv`
- `industry_revenue_distributions.csv`

and save them in a folder: `data/raw/`

#### Running the pipeline
Run the main orchestration script with command-line flags to specify which stages to execute.
Example command:
```bash
python main.py --run-data-prep --run-models --get-revenue
```

#### Available flags:
You can mix and match these flags to run only the stages you need:

```bash
--run-data-prep
```
Runs data preparation and feature validation. If omitted, cached/previously prepared data is loaded instead.
```bash
--run-models
```
Trains machine learning models using the prepared data.
```bash
--get-revenue
```
Generates predictions and calculates estimated revenue.

--- 
## Frontend Application

There is also a user interface for easily interacting with the data.
```bash
run Ml_pipeline to generate required data files before launching UI.
```



---

## Requirements
- Python 3.8+
- See requirements.txt for all dependencies

---

## Logging
Each part of the application creates its own log file which you can file in the logs folder.