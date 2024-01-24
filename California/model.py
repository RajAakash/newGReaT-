# model.py
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
from be_great import GReaT

class ModelTrainer:
    def __init__(self, llm='distilgpt2', 
                 batch_size=64, 
                 epochs=1, 
                 logging_steps=50, 
                 save_steps=400000):
        self.model = GReaT(llm=llm, 
                           batch_size=batch_size, 
                           epochs=epochs, 
                           logging_steps=logging_steps, 
                           save_steps=save_steps)

    def train(self, X_train, y_train):
        train_data = pd.concat([X_train, y_train], axis=1)
        #how can we fine tune this?
        self.model.fit(train_data)

    def generate_synthetic_data(self, num_samples):
        return self.model.sample(num_samples)

class CustomModels:
    @staticmethod
    def train_decision_tree(X_train, y_train):
        dt_model = DecisionTreeRegressor(max_depth=10,
                                         random_state=42)
        dt_model.fit(X_train, y_train)
        return dt_model

    @staticmethod
    def train_random_forest(X_train, y_train, param_grid):
        rf_model = RandomForestRegressor()
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        rf_model = RandomForestRegressor(**best_params)
        rf_model.fit(X_train, y_train)
        return rf_model, best_params

    @staticmethod
    def train_linear_regression(X_train, y_train):
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        return lr_model

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse
