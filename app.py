from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib


app = Flask(__name__)
CORS(app)


# ---------------- LOAD TRAINED MODEL ----------------

saved_data = joblib.load("house_price_model.pkl")

model = saved_data["model"]
city_map = saved_data["city_map"]
country_map = saved_data["country_map"]
feature_columns = saved_data["feature_columns"]


# ---------------- HOME PAGE ----------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICTION ----------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get data from frontend
        data = request.get_json()

        # Convert input data into DataFrame
        df = pd.DataFrame([data])


        # ---------------- DATE ----------------

        date_value = pd.to_datetime(
            df["date"],
            dayfirst=True
        )

        df["date_year"] = date_value.dt.year
        df["date_month"] = date_value.dt.month
        df["date_day"] = date_value.dt.day

        # Remove original date column
        df.drop(
            "date",
            axis=1,
            inplace=True
        )


        # ---------------- CITY ----------------

        df["city"] = df["city"].map(city_map)


        # ---------------- COUNTRY ----------------

        df["country"] = df["country"].map(country_map)


        # ---------------- FEATURE ORDER ----------------

        # Arrange columns exactly as used during training
        df = df[feature_columns]


        # ---------------- PREDICTION ----------------

        prediction = model.predict(df)


        # Send prediction to frontend
        return jsonify({
            "predicted_price": float(prediction[0])
        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# ---------------- MAIN ----------------

if __name__ == "__main__":

    print()
    print("========================================")
    print("     HOUSE PRICE PREDICTION SYSTEM")
    print("========================================")
    print()
    print("Open this link:")
    print("http://127.0.0.1:5000")
    print()
    print("========================================")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )