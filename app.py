from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("models.pkl")[1]  # RandomForest model at index 1
loaded = joblib.load("models.pkl")
# If the loaded object is a list, get the model from it
if isinstance(loaded, list):
    model = loaded[0]
else:
    model = loaded

scaler = joblib.load("scaler.pkl")

# Prediction route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Get form data and convert to float
            data = [float(request.form[field]) for field in [
                "season", "weather", "temp", "humidity", "windspeed", "casual",
                "day", "month", "year", "weekday", "am_or_pm", "holidays"
            ]]
            
            # Prepare DataFrame
            columns = ['season', 'weather', 'temp', 'humidity', 'windspeed', 'casual',
                       'day', 'month', 'year', 'weekday', 'am_or_pm', 'holidays']
            input_df = pd.DataFrame([data], columns=columns)
            
            # Scale input
            scaled_input = scaler.transform(input_df)
            
            # Predict
            result = model.predict(scaled_input)
            prediction = f"Predicted Ride Demand: {round(result[0], 2)}"
        except Exception as e:
            prediction = f"Error: {e}"
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
