import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib 

# Load Dataset
data = pd.read_csv("Data/Salary_Data.csv")
print("Dataset head:\n", data.head())

# Split Features & Target
X = data[['YearsExperience']]
y = data['Salary']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction 
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Visualization
plt.scatter(X_test, y_test, color='blue', label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction")
plt.legend()
plt.show()

# Save Model
joblib.dump(model, "salary_model.pkl")
print("Model saved as salary_model.pkl")