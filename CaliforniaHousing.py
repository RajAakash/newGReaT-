from sklearn.datasets import fetch_california_housing,load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV
from be_great import GReaT
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = fetch_california_housing(as_frame=True).frame 
model = GReaT(llm='distilgpt2', batch_size=64, epochs=5, logging_steps=50,save_steps=400000)
model.fit(data)
synthetic_data = model.sample(n_samples=100)
data = fetch_california_housing(as_frame=True).frame 
X = synthetic_data.drop(columns=['MedHouseVal'])
y=synthetic_data['MedHouseVal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

# Decision Tree
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_predictions)
print(f"Decision Tree Mean Squared Error: {dt_mse}")

# Random Forest
rf_model = RandomForestRegressor()

#Search with cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X, y)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Create a Linear Regression model with custom hyperparameters
custom_lr_model = LinearRegression(fit_intercept=False)  # Adjust hyperparameters as needed

# Fit the model on the training data
custom_lr_model.fit(X_train, y_train)
# Make predictions on the test set
lr_predictions_custom = custom_lr_model.predict(X_test)
# Calculate Mean Squared Error
lr_mse_custom = mean_squared_error(y_test, lr_predictions_custom)
print(f"Custom Linear Regression Mean Squared Error: {lr_mse_custom}")

best_params = {'max_depth': None, 'min_samples_leaf': 10, 'n_estimators': 100}
rf_model = RandomForestRegressor(**best_params)
# Fit the model on the training data
rf_model.fit(X_train, y_train)
# Make predictions on the test set
predictions = rf_model.predict(X_test)
# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error with Best Hyperparameters: {mse}")

