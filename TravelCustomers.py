from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from be_great import GReaT
import pandas as pd
import os,pickle
from sklearn.preprocessing import LabelEncoder

#real_data = load_diabetes(as_frame=True).frame
real_data=pd.read_csv('Customertravel.csv')
label_encoder = LabelEncoder()
real_data['AccountSyncedToSocialMedia'] = real_data['AccountSyncedToSocialMedia'].map({'Yes': 1, 'No': 0})
real_data['FrequentFlyer'] = real_data['FrequentFlyer'].map({'Yes': 1, 'No': 0,'No Record':-1})
real_data['AnnualIncomeClass'] = real_data['AnnualIncomeClass'].map({'Low Income': 0, 'Middle Income': 1,'High Income':2})
real_data=real_data.drop(columns=['ServicesOpted'])
real_data['BookedHotelOrNot'] = real_data['BookedHotelOrNot'].map({'Yes': 1, 'No': 0})
print(real_data.head(5))
X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
    real_data.drop(columns=['Target']),real_data['Target'], test_size=0.2, random_state=42
)

label_encoder = LabelEncoder()

if os.path.isfile('travel.pkl'):
  with open('travel.pkl','rb') as infile:
    model = pickle.load(infile)
else:
  model = GReaT(llm='distilgpt2', batch_size=124, epochs=100,
                logging_steps=50,save_steps=400000)
  # Fit the model on synthetic data
  model.fit(real_data)
  with open('travel.pkl','wb') as outfile:
    pickle.dump(model, outfile)

# Initialize your GReaT model
model = GReaT(llm='distilgpt2', batch_size=64, epochs=100, logging_steps=50, save_steps=400000)

real_train_data = pd.concat([X_real_train, y_real_train], axis=1)

# Fit the model on synthetic data
model.fit(real_train_data)

# Generate synthetic data
synthetic_data = model.sample(n_samples=X_real_train.shape[0])

# Separate features and target variable
X_synthetic = synthetic_data.drop(columns=['Target'])
y_synthetic = synthetic_data['Target']

# Use synthetic data for training
X_train = X_synthetic
y_train = y_synthetic
y_train_encoded=label_encoder.fit_transform(y_train)

# Use real data for testing
X_test = X_real_test
y_test = y_real_test
y_test_encoded=label_encoder.transform(y_test)
#if os.path.isfile('.pkl'):
#  with open('new.pkl','rb') as infile:
#    model = pickle.load(infile)
#else:
#  model = GReaT(llm='distilgpt2', batch_size=124, epochs=100,
#                logging_steps=50,save_steps=400000)
  # Fit the model on synthetic data
#  model.fit(real_data)
#  with open('new.pkl','wb') as outfile:
#    pickle.dump(model, outfile)
#real_data = load_diabetes(as_frame=True).frame

accuracies = {'Model': [], 'Accuracy': []}
logistic_regression_model = LogisticRegression()
print(f'{y_train_encoded.shape}-{X_train.shape} here ')
logistic_regression_model.fit(X_train, y_train_encoded)
logistic_regression_predictions = logistic_regression_model.predict(X_test)

# Decision Tree
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train_encoded)
decision_tree_predictions = decision_tree_model.predict(X_test)

# Random Forest
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train_encoded)
random_forest_predictions = random_forest_model.predict(X_test)

def evaluate_model(model_name, predictions):
    accuracy = accuracy_score(y_test_encoded, predictions)
    print(f"------ {model_name} Model ------")
    accuracies['Model'].append(model_name)
    accuracies['Accuracy'].append(accuracy)
    print(f"Accuracy: {accuracy:.2f}")

# Display evaluation results
evaluate_model("Logistic Regression", logistic_regression_predictions)
evaluate_model("Decision Tree", decision_tree_predictions)
evaluate_model("Random Forest", random_forest_predictions)
print(accuracies)
data={
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Mean Squared Error': [logistic_regression_predictions, decision_tree_predictions, random_forest_predictions]
}
df=pd.DataFrame(accuracies)
df.to_csv('mse_results_travel.csv',index=False)
print('Results saved to csv file')
