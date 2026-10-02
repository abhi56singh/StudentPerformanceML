import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Input features
X = df[["studytime", "failures", "absences", "G1", "G2"]]

# Target
y = df["G3"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RANDOM FOREST MODEL")
print("-------------------------")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# Save model
with open("student_performance_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")
import matplotlib.pyplot as plt

# -----------------------------
# Actual vs Predicted Graph
# -----------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Final Grade (G3)")
plt.ylabel("Predicted Final Grade (G3)")
plt.title("Actual vs Predicted Final Grades")

# Perfect prediction reference line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.tight_layout()

plt.savefig("actual_vs_predicted.png")

plt.show()

print("\nGraph saved successfully!")