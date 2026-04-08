import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("sales_data.csv")

# Features and target
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "sales_model.pkl")
print("Model trained and saved as sales_model.pkl")