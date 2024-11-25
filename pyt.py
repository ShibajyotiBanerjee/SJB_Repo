import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    'Cerebral Blood Flow Velocity': np.random.randn(1000),
    'Cerebral Blood Volume': np.random.randn(1000),
    'Cerebral Metabolic Rate of Oxygen': np.random.randn(1000),
    'Tissue Oxygen Tension': np.random.randn(1000),
    'Jugular Venous Oxygen Saturation': np.random.randn(1000),
    'Cerebral Oxygenation Level': np.random.randint(2, size=1000)  # 0 for low, 1 for high
}

# Create a Pandas DataFrame
df = pd.DataFrame(data)

# Split data into features (X) and target variable (y)
X = df.drop('Cerebral Oxygenation Level', axis=1)
y = df['Cerebral Oxygenation Level']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

client = mlflow.tracking.MlflowClient()

mlflow.tracking.MlflowClient()

# Start an MLflow experiment
mlflow.set_experiment("Oxygenation")

with mlflow.start_run() as run:
    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters (optional)
    mlflow.log_param("model_name", "Logistic Regression")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(model, "cerebral_oxygenation_model")

    # Get the run ID
    run_id = run.info.run_id

    print (f"Run ID is: {run_id}")

# Load the logged model
loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/Cerebral_oxygenation_model")

# Function to predict cerebral oxygenation level
def predict_cerebral_oxygenation(model, features):
    prediction = model.predict([features])
    if prediction == 1:
        return "High"
    else:
        return "Low"

# Get user input for each feature
features = []
for feature_name in X.columns:
    value = float(input(f"Enter {feature_name}: "))
    features.append(value)

# Predict cerebral oxygenation level
predicted_level = predict_cerebral_oxygenation(loaded_model, features)

print("Predicted Cerebral Oxygenation Level:", predicted_level)