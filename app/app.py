# ANA68-Midterm app.py
# app/app.py

from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("app/model.pkl")
label_encoder = joblib.load("app/label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        math = float(request.form["math_score"])
        reading = float(request.form["reading_score"])
        writing = float(request.form["writing_score"])

        features = np.array([[math, reading, writing]])
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        return render_template("index.html", prediction_text=f"Predicted Race/Ethnicity: {predicted_label}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)