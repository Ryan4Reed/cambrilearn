# Cambrilearn project

This project is designed to make the best use of our claims data in order to predict the expected revenue post `2024-06-30`

We have also implemented a simple user interface for interacting with the data on an account level.

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
- See the **Frontend Application** section below.
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
├── documentation/
│   └── report.md
├── main.py
├── requirements.txt
├── app/
│   ├── app.py
└── ├── templates/index.html
```

## Additional documentation
Please find additional documentation wrt the notes taken, considerations and decisions made in the documentation folder.

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
You will need to run the ml pipeline in order to generate required data files before you can make use of the UI.

#### To launch UI
Simply execute the following command in termninal and head to `http://127.0.0.1:5000/`
```bash
python app/app.py
```

---

## Requirements
- Python 3.8+
- See requirements.txt for all dependencies

---

## Logging
Each part of the application creates its own log file which you can file in the logs folder.