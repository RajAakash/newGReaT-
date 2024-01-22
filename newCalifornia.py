from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from be_great import GReaT
import pandas as pd
import os, pickle

def traintest(X_train, X_test, y_train, y_test):
  mse = {}
  for model in [('LR',LinearRegression()),
                ('DT',DecisionTreeRegressor(max_depth=10,
                                            random_state=42)),
                ('RF',RandomForestRegressor(n_estimators=85,
                                            max_depth=12,
                                            random_state=42))]:
    model[1].fit(X_train, y_train)
    predictions = model[1].predict(X_test)
    mse[model[0]] = mean_squared_error(y_test, predictions)
    print(f'{model[0]} MSE: {mse[model[0]]}')
  return mse

real_data = fetch_california_housing(as_frame=True).frame
X_train, X_test, y_train, y_test = \
                  train_test_split(real_data.drop(columns=['MedHouseVal']),
                                    real_data['MedHouseVal'],
                                    test_size=0.2, random_state=42)


real_data = pd.concat([X_train, y_train], axis=1)
print(f'real_data\n{real_data.describe()}')
traintest(X_train, X_test, y_train, y_test)

# Get GReaT model
if os.path.isfile('great.pkl'):
  with open('great.pkl','rb') as infile:
    model = pickle.load(infile)
else:
  model = GReaT(llm='distilgpt2', batch_size=124, epochs=100,
                logging_steps=50,save_steps=400000)
  # Fit the model on synthetic data
  model.fit(real_data)
  with open('great.pkl','wb') as outfile:
    pickle.dump(model, outfile)

print('====================')

# Generate synthetic data
synthetic_data = model.sample(n_samples=real_data.shape[0])
print(f'synthetic_data\n{synthetic_data.describe()}')

# Use synthetic data for training
X_train = synthetic_data.drop(columns=['MedHouseVal'])
y_train = synthetic_data['MedHouseVal']

mse = traintest(X_train, X_test, y_train, y_test)

with open('mse_results.csv','w') as outfile:
  outfile.write('Model,MSE\n')
  for model in mse:
    outfile.write(f'{model},{mse[model]}\n')

print('MSE values saved to mse_results.csv')
