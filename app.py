from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("sales_model.pkl")

# Load dataset and calculate predictions for historical data
data = pd.read_csv("sales_data.csv")
data['Predicted_Sales'] = model.predict(data[['TV', 'Radio', 'Newspaper']])

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    chart_data = None

    if request.method == "POST":
        try:
            # Get input values
            TV = float(request.form["TV"])
            Radio = float(request.form["Radio"])
            Newspaper = float(request.form["Newspaper"])

            if TV < 0 or Radio < 0 or Newspaper < 0:
                raise ValueError("Values cannot be negative.")

            # Predict for user input
            df = pd.DataFrame([[TV, Radio, Newspaper]], columns=["TV", "Radio", "Newspaper"])
            prediction = round(float(model.predict(df)[0]), 2)

            # Prepare chart data including historical and user input
            chart_data = {
                "labels": [f"Record {i+1}" for i in range(len(data))] + ["Your Input"],
                "actual": list(data['Sales']) + [prediction],        # User input point same as prediction
                "predicted": list(data['Predicted_Sales']) + [prediction]
            }

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error, chart_data=chart_data)

if __name__ == "__main__":
    app.run(debug=True)