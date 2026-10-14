# Telco Customer Churn Prediction

## Description
This project uses Machine Learning to predict whether a telecom customer is likely to leave the company.

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- MLflow
- Streamlit

## Features
- Data analysis and preprocessing.
- Customer segmentation using K-Means.
- Training and comparison of classification models.
- Churn prediction using a Streamlit application.

## Prediction Target
The target variable is `Churn`, which indicates whether a customer has left the company.

- `Yes`: The customer left the company.
- `No`: The customer stayed with the company.

## Target vs Features
- **Target (`y`):** `Churn`, the variable we want to predict.
- **Features (`X`):** Customer information used for prediction, such as `tenure`, `MonthlyCharges`, `Contract`, and `InternetService`.

The model learns from the features to predict whether a customer will leave the company.

## Churn Encoding
The target can be encoded numerically for Machine Learning:

- `Yes` → `1`
- `No` → `0`

The actual encoding depends on the preprocessing used during model training.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

