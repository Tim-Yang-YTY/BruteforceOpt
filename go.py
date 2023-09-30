# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Read the data (assuming it's in a CSV format)
data = pd.read_csv('your_data_file.csv')

# Handle missing values (you might want to handle them differently based on your understanding of the data)
data.fillna(method='ffill', inplace=True)  # Forward fill as a simple imputation method

# Convert categorical features to dummy variables (one-hot encoding)
data = pd.get_dummies(data)

# Split the data into training and test sets (70% training, 30% test)
X = data.drop('RECL_COUT_REPARATION_NUM', axis=1)  # Features (excluding the target)
y = data['RECL_COUT_REPARATION_NUM']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# If you want to see feature importance
feature_importance = rf.feature_importances_
for i, imp in enumerate(feature_importance):
    print(f"Feature {X.columns[i]}: {imp}")
