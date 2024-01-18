from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, accuracy_score
from be_great import GReaT

diabetes_data=load_diabetes(as_frame=True).frame

model = GReaT(llm='distilgpt2', batch_size=64, epochs=5, logging_steps=50,save_steps=400000)
model.fit(diabetes_data)
synthetic_data = model.sample(n_samples=100)
X=synthetic_data.drop(columns=['target'])
y=synthetic_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
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

# Evaluate models
def evaluate_model(model_name, predictions):
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"------ {model_name} Model ------")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)

# Display evaluation results
evaluate_model("Logistic Regression", logistic_regression_predictions)
evaluate_model("Decision Tree", decision_tree_predictions)
evaluate_model("Random Forest", random_forest_predictions)

