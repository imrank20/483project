# House Price Prediction Project

## Overview
This project predicts house prices based on features like square footage, number of bedrooms, bathrooms, and other factors.

## Files
- `main.py`: Main Python script that loads data, trains models, and evaluates them.
- `requirements.txt`: Python dependencies required to run the project.

## Steps to Run
1. Install dependencies using:
    pip install -r requirements.txt

2. Download the dataset from Kaggle ([King County House Prices](https://www.kaggle.com/harlfoxem/housesalesprediction)) and place the `kc_house_data.csv` file in the project directory.

3. Run the main script:
    python main.py


## Models Used
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Each model is evaluated using Mean Squared Error (MSE) and R-squared metrics.
