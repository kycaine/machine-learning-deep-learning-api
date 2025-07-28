# ML-DL General API Project
This project is a general-purpose machine learning / deep learning API built with FastAPI.

It allows users to upload any dataset (CSV/Excel), configure task parameters, and automatically run through a full ML pipeline — including cleaning, EDA, feature engineering, training, prediction, and visualization.

# Project Structure

ML-DL-API-PROJECT/
├── app/
│ ├── api/
│ │ ├── router/
│ │ │ ├── machine_learning_routes.py
│ │ │ └── download_routes.py
│ │ └── schemas/
│ │ └── general_request.py
│ ├── config/
│ │ ├── file_manager.py
│ │ └── constants.py
│ ├── machine_learning/
│ │ └── preprocessing.py
│ │ └── eda.py
│ │ └── processing.py
│ └── main.py
├── data/
├── docs/
├── models/
├── outputs
├── .gitignore
├── README.md
└── requirements.txt

# API
| Endpoint                    | Description                                          |
| --------------------------- | ---------------------------------------------------- |
| `POST /clean`               | Clean raw dataset (handle missing values, fix types) |
| `POST /eda`                 | Generate automated visualizations and basic stats    |
| `POST /feature-engineering` | Apply feature transformations                        |
| `POST /train-and-predict`   | Train model and return predictions & visualizations  |
| `GET /download/{file_path}` | Download result files (CSV, ZIP, images, etc.)       |


# INPUT (API Request Body)

basic payload metadata

{
  "target_column": "string",                    // Column you want to predict
  "task_type": "regression/classification",     // Choose the type of task
  "feature_columns": ["col1", "col2", ...],     // List of input features
  "columns": {                                  // Column names and data types
    "col1": "str/int/float",
    "col2": "str/int/float"
  }
}


Example 1: payload raw dataset
{
  "target_column": "new_cases",
  "task_type": "regression",
  "feature_columns": ["date", "country", "new_deaths", "total_cases", "total_deaths", "recovered", "active_cases", "vaccinated"],
  "columns": {
    "id": "int",
    "date": "str",
    "country": "str",
    "new_cases": "float",
    "new_deaths": "float",
    "total_cases": "float",
    "total_deaths": "float",
    "recovered": "float",
    "active_cases": "float",
    "vaccinated": "float"
  }
}


Example 2: after feature enginnering
Once your data goes through feature engineering (e.g., date breakdowns, country encoding), your input may look like this:

{
  "target_column": "total_cases",
  "task_type": "regression",
  "feature_columns": [
    "new_deaths",
    "total_cases",
    "total_deaths",
    "recovered",
    "active_cases",
    "vaccinated",
    "date_dayofweek",
    "date_month",
    "date_day",
    "country_Germany",
    "country_India",
    "country_Italy",
    "country_Spain",
    "country_USA"
  ],
  "columns": {
    "id": "int",
    "date": "str",
    "country": "str",
    "new_cases": "float",
    "new_deaths": "float",
    "total_cases": "float",
    "total_deaths": "float",
    "recovered": "float",
    "active_cases": "float",
    "vaccinated": "float",
    "date_dayofweek": "int",
    "date_month": "int",
    "date_day": "int",
    "country_Germany": "int",
    "country_India": "int",
    "country_Italy": "int",
    "country_Spain": "int",
    "country_USA": "int"
  }
}



# I/O WORK FLOW
| Step                 | Input                      | Output                                |
| -------------------- | -------------------------- | ------------------------------------- |
| Clean Data           | Raw CSV                    | Cleaned DataFrame                     |
| Exploratory Analysis | Cleaned DataFrame          | Stats, plots                          |
| Feature Engineering  | Cleaned DataFrame          | Transformed features                  |
| Train + Predict      | Engineered data + metadata | Predictions, metrics, plots, CSV, ZIP |
| Download File        | File path                  | Actual CSV / ZIP                      |


# End-to-End ML Process

Data Cleaning → Fix missing values, cast types

EDA → Understand patterns and distributions

Feature Engineering → Create enhanced inputs for the model

Modeling → Train RandomForest (or similar)

Model Evaluation → View MAE, RMSE, R² metrics

Visualization → Plot predictions vs actuals

Download → Get outputs (CSV/ZIP)
