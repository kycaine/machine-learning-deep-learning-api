# MAchine Learning
## Data cleaning, EDA and feture engineering API request
```json
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
```

## train and predict API request
```json
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
    "country_Canada",
    "country_France",
    "country_Germany",
    "country_India",
    "country_Japan",
    "country_UK",
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
    "country_Canada": "int",
    "country_France": "int",
    "country_Germany": "int",
    "country_India": "int",
    "country_Japan": "int",
    "country_UK": "int",
    "country_USA": "int"
  }
}
```
