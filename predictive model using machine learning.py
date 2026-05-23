# ============================================
# PREDICTIVE MODELING USING MACHINE LEARNING
# ============================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------
# STEP 1: LOAD DATASET
# --------------------------------------------

# Replace with your CSV file name
data = pd.read_csv("student-mat.csv", sep=';')

# Display first 5 rows
print("Dataset Preview:")
print(data.head())

# --------------------------------------------
# STEP 2: SELECT FEATURES AND TARGET
# --------------------------------------------

# Input Features
X = data[['studytime', 'failures', 'absences']]

# Target Variable
y = data['G3']   # Final Grade

# --------------------------------------------
# STEP 3: SPLIT DATA
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------
# STEP 4: TRAIN MODEL
# --------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# --------------------------------------------
# STEP 5: MAKE PREDICTIONS
# --------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------
# STEP 6: EVALUATE MODEL
# --------------------------------------------

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-------------------")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# --------------------------------------------
# STEP 7: VISUALIZATION
# --------------------------------------------

plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")

plt.title("Actual vs Predicted Grades")

plt.show()

# --------------------------------------------
# STEP 8: PREDICT NEW DATA
# --------------------------------------------

# Example:
# studytime = 3
# failures = 0
# absences = 4

new_data = pd.DataFrame({
    'studytime': [3],
    'failures': [0],
    'absences': [4]
})

prediction = model.predict(new_data)

prediction = model.predict(new_data)

print("\nPredicted Final Grade:", prediction[0])
