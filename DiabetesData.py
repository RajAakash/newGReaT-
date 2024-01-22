from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from be_great import GReaT
import pandas as pd

#real_data = load_diabetes(as_frame=True).frame
real_data=pd.read_csv('diabetes.csv')
print(real_data.head(5))
X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
    real_data.drop(columns=['Outcome']),real_data['Outcome'], test_size=0.2, random_state=42
)
# Initialize your GReaT model
model = GReaT(llm='distilgpt2', batch_size=64, epochs=100, logging_steps=50, save_steps=400000)

real_train_data = pd.concat([X_real_train, y_real_train], axis=1)

# Fit the model on synthetic data
model.fit(real_train_data)

# Generate synthetic data
synthetic_data = model.sample(n_samples=X_real_train.shape[0])
print("====================")
print(synthetic_data.head(5))
print("====================")

# Separate features and target variable
X_synthetic = synthetic_data.drop(columns=['Outcome'])
y_synthetic = synthetic_data['Outcome']

# Use synthetic data for training
X_train = X_synthetic
y_train = y_synthetic

# Use real data for testing
X_test = X_real_test
y_test = y_real_test

accuracies = {'Model': [], 'Accuracy': []}
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)
logistic_regression_predictions = logistic_regression_model.predict(X_test)

# Decision Tree
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)
decision_tree_predictions = decision_tree_model.predict(X_test)

# Random Forest
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)
random_forest_predictions = random_forest_model.predict(X_test)

def evaluate_model(model_name, predictions):
    accuracy = accuracy_score(y_test, predictions)
    print(f"------ {model_name} Model ------")
    accuracies['Model'].append(model_name)
    accuracies['Accuracy'].append(accuracy)
    print(f"Accuracy: {accuracy:.2f}")

# Display evaluation results
evaluate_model("Logistic Regression", logistic_regression_predictions)
evaluate_model("Decision Tree", decision_tree_predictions)
evaluate_model("Random Forest", random_forest_predictions)
print(accuracies)
data=
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Mean Squared Error': [logistic_regression_predictions, decision_tree_predictions, random_forest_predictions]
}
df=pd.DataFrame(accuracies)
df.to_csv('mse_results_diabetes.csv',index=False)
print('Results saved to csv file')
