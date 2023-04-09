import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from cfel import CFEL


def test_function():
    # Step 1: Load and preprocess data
    data = pd.read_csv('data.csv')

    # Encode the categorical variables
    le = LabelEncoder()

    data['age'] = le.fit_transform(data['age'])
    data['income'] = le.fit_transform(data['income'])
    data['education'] = le.fit_transform(data['education'])
    data['employment'] = le.fit_transform(data['employment'])
    data['marital_status'] = le.fit_transform(data['marital_status'])
    data['gender'] = le.fit_transform(data['gender'])

    X = data.drop('marital_status', axis=1)
    y = data['marital_status']

    # Step 2: Train machine learning model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    cfel = CFEL(X_train,model)
    # Step 3: Counterfactual explanations
    cf = cfel.find_counterfactuals( X_train, y_train)
    x = X_test.iloc[0]
    target_class = 0  # The desired class for the counterfactual example
    cfel.plot_counterfactual( cf)

    # Step 4: Shapley values
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)

    # Step 5: Local surrogate models
    n_neighbors = 10
    sample = X_test.iloc[0]
    knn = LinearRegression()
    knn.fit(X_train.iloc[:n_neighbors], y_train.iloc[:n_neighbors])
    prediction = knn.predict(sample.values.reshape(1, -1))
    print(f"KNN prediction: {prediction[0]}")

    # Step 6: Contrastive explanations
    x0 = X_test.iloc[0]
    x1 = X_test.iloc[1]
    delta = x1 - x0
    print(f"Contrastive explanation for x1 - x0: {delta}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_function()
