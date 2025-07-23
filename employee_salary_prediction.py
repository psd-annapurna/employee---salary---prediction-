# employee_salary_prediction.py

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load sample dataset (you can replace it with your own)
data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/Salary%20Data.csv")

# Display first few rows
print("Dataset Preview:")
print(data.head())

# Check for null values
print("\nMissing Values:\n", data.isnull().sum())

# Visualize the data
sns.scatterplot(x='YearsExperience', y='Salary', data=data)
plt.title("Experience vs Salary")
plt.show()

# Prepare data
X = data[['YearsExperience']]
y = data['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Coefficients:")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("\nModel Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Predict salary for 5 years of experience
experience = [[5]]
predicted_salary = model.predict(experience)
print(f"\nPredicted Salary for 5 years of experience: ₹{predicted_salary[0]:.2f}")