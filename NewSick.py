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
real_data=pd.read_csv('dataset_38_sick.csv')
label_encoder = LabelEncoder()
columns_to_convert = ['on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication','sick',
        'pregnant','thyroid_surgery','I131_treatment','query_hypothyroid','query_hyperthyroid',
        'lithium','goitre','tumor','hypopituitary','psych','TSH_measured','T3_measured','TT4_measured',
        'T4U_measured','FTI_measured','TBG_measured']  # List the columns you want to convert
real_data[columns_to_convert] = real_data[columns_to_convert].replace({'t': 1, 'f': 0})
most_frequent_gender = real_data['sex'].mode()[0]
real_data['Class']=real_data['Class'].map({'negative': 0, 'sick': 1})
real_data['age'].fillna(0, inplace=True)
real_data['sex']=real_data['sex'].map({'F': 1, 'M': 0})
real_data['TSH'] = pd.to_numeric(real_data['TSH'], errors='coerce')
real_data['TSH'].fillna(real_data['TSH'].mean(), inplace=True)

real_data['T3'] = pd.to_numeric(real_data['T3'], errors='coerce')
real_data['T3'].fillna(real_data['T3'].mean(), inplace=True)

real_data['TT4'] = pd.to_numeric(real_data['TT4'], errors='coerce')
real_data['TT4'].fillna(real_data['TT4'].mean(), inplace=True)

real_data['T4U'] = pd.to_numeric(real_data['T4U'], errors='coerce')
real_data['T4U'].fillna(real_data['T4U'].mean(), inplace=True)

real_data['FTI'] = pd.to_numeric(real_data['FTI'], errors='coerce')
real_data['FTI'].fillna(real_data['FTI'].mean(), inplace=True)

real_data=real_data.drop(columns=['TBG'])
real_data=real_data.drop(columns=['referral_source'])

print(real_data.head(5))

X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
    real_data.drop(columns=['Class']),real_data['Class'], test_size=0.2, random_state=42
)

label_encoder = LabelEncoder()

if os.path.isfile('sick.pkl'):
  with open('sick.pkl','rb') as infile:
    model = pickle.load(infile)
else:
  model = GReaT(llm='distilgpt2', batch_size=124, epochs=100,
                logging_steps=50,save_steps=400000)
  # Fit the model on synthetic data
  model.fit(real_data)
  with open('sick.pkl','wb') as outfile:
    pickle.dump(model, outfile)

# Initialize your GReaT model
model = GReaT(llm='distilgpt2', batch_size=64, epochs=100, logging_steps=50, save_steps=400000)

real_train_data = pd.concat([X_real_train, y_real_train], axis=1)

# Fit the model on synthetic data
model.fit(real_train_data)

# Generate synthetic data
synthetic_data = model.sample(n_samples=X_real_train.shape[0])

# Separate features and target variable
X_synthetic = synthetic_data.drop(columns=['Class'])
y_synthetic = synthetic_data['Class']

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
