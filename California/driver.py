# driver.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model import ModelTrainer, CustomModels
import pandas as pd
from config import filename, target_column,text_columns_to_encode,epochs,batch_size

real_data = pd.read_csv(filename)

# Split data into train and test sets
X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
    real_data.drop(columns=[target_column]), 
    real_data[target_column], test_size=0.2, random_state=42
)

print(f'{real_data.head(5)}')
model_trainer = ModelTrainer(batch_size=batch_size,epochs=epochs)
model_trainer.train(X_real_train, y_real_train)
synthetic_data = model_trainer.generate_synthetic_data(X_real_train.shape[0])

X_synthetic = synthetic_data.drop(columns=[target_column])
y_synthetic = synthetic_data[target_column]

label_encoder = LabelEncoder()
for column in text_columns_to_encode:
    if column in real_data.columns:
        real_data[column] = label_encoder.fit_transform(real_data[column])

# Use synthetic data for training
X_train = X_synthetic
y_train = y_synthetic

# Use real data for testing
X_test = X_real_test
y_test = y_real_test

# Train and evaluate models
dt_model = CustomModels.train_decision_tree(X_train, y_train)
dt_mse = CustomModels.evaluate_model(dt_model, X_test, y_test)
print(f"Decision Tree Mean Squared Error: {dt_mse}")

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

rf_model, best_params = CustomModels.train_random_forest(X_train, y_train, param_grid)
rf_mse = CustomModels.evaluate_model(rf_model, X_test, y_test)
print(f"Random Forest Mean Squared Error: {rf_mse}")

lr_model = CustomModels.train_linear_regression(X_train, y_train)
lr_mse = CustomModels.evaluate_model(lr_model, X_test, y_test)
print(f"Custom Linear Regression Mean Squared Error: {lr_mse}")

# Save the MSE results to a CSV file
mse_data = {
    'Model': ['Decision Tree', 'Random Forest', 'Linear Regression'],
    'Mean Squared Error': [dt_mse, rf_mse, lr_mse]
}

mse_df = pd.DataFrame(mse_data)
mse_df.to_csv('mse_results.csv', index=False)
print("MSE values saved to mse_results.csv")
