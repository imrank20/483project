import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_and_preprocess_data():
    # Load the data
    df = pd.read_csv('kc_house_data.csv')

    # Drop irrelevant features
    df = df.drop(['id', 'date'], axis=1)

    # Feature Engineering (e.g., age of the house)
    df['age'] = 2024 - df['yr_built']

    # Train/Test Split
    X = df.drop('price', axis=1)  # Features
    y = df['price']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse}, R2: {r2}")

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    evaluate_model(y_test, y_pred_lr, 'Linear Regression')

    # Decision Tree Regressor
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    evaluate_model(y_test, y_pred_dt, 'Decision Tree')

    # Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    evaluate_model(y_test, y_pred_rf, 'Random Forest')

if __name__ == '__main__':
    main()
 