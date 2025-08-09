# train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn import metrics

# 1. Load dataset
car_dataset = pd.read_csv("car data.csv")  # Ensure the file is in the same folder

# 2. Encode categorical variables
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
car_dataset.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

# 3. Split into features (X) and target (Y)
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

# 4. Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

# 5. Train Lasso Regression model
lass_reg = Lasso()
lass_reg.fit(X_train, Y_train)

# 6. Evaluate model
train_pred = lass_reg.predict(X_train)
train_score = metrics.r2_score(Y_train, train_pred)
print(f"Training R² Score: {train_score:.4f}")

test_pred = lass_reg.predict(X_test)
test_score = metrics.r2_score(Y_test, test_pred)
print(f"Test R² Score: {test_score:.4f}")

# 7. Save model
with open("lasso_model.pkl", "wb") as f:
    pickle.dump(lass_reg, f)

print("✅ Model saved as lasso_model.pkl")
