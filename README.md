# ML-DL General API Project

"This project is a general machine learning / deep learning API.  
Users can upload any dataset (CSV/Excel) and define:

- Which column to predict (`target_column`)
- Task type: `regression` or `classification`
- Which columns to use as features

The pipeline will automatically train and evaluate the model."

# Project Structure

ML-DL-API-PROJECT/
├── app/
│ ├── api/
│ │ ├── router/
│ │ │ ├── data_cleaning_routes.py
│ │ │ └── download_routes.py
│ │ └── schemas/
│ │ └── schemas.py
│ ├── config/
│ │ ├── init_folders.py
│ │ └── paths.py
│ ├── machine_learning/
│ │ └── preprocessing.py
│ └── main.py
├── data/
├── docs/
├── models/
├── .gitignore
├── README.md
└── requirements.txt

# INPUT (API Request Body)

```json
{
  "target_column": "price",
  "task_type": "regression",
  "feature_columns": ["size", "rooms", "location"],
  "columns": {
    "price": "float",
    "size": "float",
    "rooms": "int",
    "location": "str"
  }
}
```

# OUTPUT

```json
{
  "status": "success",
  "score": 0.87,
  "model_path": "models/model_123.pkl",
  "feature_plot": "outputs/plot_123.png"
}
```

# How it works

Upload dataset (.csv or .xlsx)

Send JSON request with task and target info

The API processes, trains, and evaluates automatically

Receive back the model, score, and plots

# I/O WORK FLOW

Data Cleaning
Input : Raw data
Output : Cleaned data

Exploratory Data Analysis (EDA)
Input : Output from data cleaning
Output : Visualization & data statistics

Feature Engineering
Input : Output from data cleaning
Output : Transformed features to enhance the model

Modeling
Input : Output from feature engineering
Output : The predictions

Model Evaluation
Input : The predictions from modeling
Output : Assessment of model performance

Model Optimization
Input:

- Output from model evaluation

- The predictions from modeling

Output : Optimized model performance

# End-to-End ML Process

Data Cleaning → Removes noise, handles missing values, and corrects errors in the dataset.

Exploratory Data Analysis (EDA) → Visualizes and analyzes data patterns to gain insights.

Feature Engineering → Creates new features or transforms existing ones to improve model performance.

Modeling → Trains a machine learning model using the processed data.

Model Evaluation → Measures model performance using metrics such as MAE, RMSE, R², or accuracy.

Model Optimization → Enhances model performance through hyperparameter tuning or other optimization techniques.
