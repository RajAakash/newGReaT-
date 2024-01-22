from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV
from be_great import GReaT
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

real_data = fetch_california_housing(as_frame=True).frame
print(real_data.head(5))

X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
    real_data.drop(columns=['MedHouseVal']),real_data['MedHouseVal'], test_size=0.2, random_state=42
)
print(X_real_train.shape[0])
# Initialize your GReaT model
model = GReaT(llm='distilgpt2', batch_size=64, epochs=1, logging_steps=50,save_steps=400000)

real_train_data = pd.concat([X_real_train, y_real_train], axis=1)

# Fit the model on synthetic data
model.fit(real_train_data)

# Generate synthetic data
synthetic_data = model.sample(X_real_train.shape[0])
print("====================")
print(synthetic_data.head(5))
print("====================")

# Separate features and target variable
X_synthetic = synthetic_data.drop(columns=['MedHouseVal'])
y_synthetic = synthetic_data['MedHouseVal']

# Use synthetic data for training
X_train = X_synthetic
y_train = y_synthetic

# Use real data for testing
X_test = X_real_test
y_test = y_real_test


param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

# Decision Tree
dt_model = DecisionTreeRegressor()
#dt_model.fit(X_real_train, y_real_train)
#dt_predictions = dt_model.predict(X_test)
dt_model.fit(X_train,y_train)
dt_predictions=dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_predictions)
print(f"Decision Tree Mean Squared Error: {dt_mse}")

# Random Forest
rf_model = RandomForestRegressor()

#Search with cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

lr_model = LinearRegression()  
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_predictions)
print(f"Custom Linear Regression Mean Squared Error: {lr_mse}")

best_params = {'max_depth': None, 'min_samples_leaf': 10, 'n_estimators': 100}
rf_model = RandomForestRegressor(**best_params)
rf_model.fit(X_train, y_train)
predictions = rf_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error with Best Hyperparameters: {mse}")

# Creating a DataFrame with the MSE values
data = {
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest'],
    'Mean Squared Error': [lr_mse, dt_mse, mse]
}

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('mse_results1.csv', index=False)

print("MSE values saved to mse_results.csv")


