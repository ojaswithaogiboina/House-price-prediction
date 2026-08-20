from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os


app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "house_price_model.pkl"
)


try:

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        saved_data = pickle.load(file)

    model = saved_data["model"]

    city_map = saved_data["city_map"]

    country_map = saved_data["country_map"]

    feature_columns = saved_data["feature_columns"]

    print("Model loaded successfully!")

except Exception as e:

    print("Error loading model:", e)

    model = None


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        if model is None:

            return jsonify({
                "error":
                "Model could not be loaded."
            }), 500


        data = request.get_json()


        if not data:

            return jsonify({
                "error":
                "No input data received."
            }), 400


        required_fields = [

            "date",

            "bedrooms",

            "bathrooms",

            "sqft_living",

            "sqft_lot",

            "floors",

            "waterfront",

            "view",

            "condition",

            "sqft_above",

            "sqft_basement",

            "yr_built",

            "yr_renovated",

            "city",

            "country"
        ]


        for field in required_fields:

            if field not in data:

                return jsonify({
                    "error":
                    f"Missing field: {field}"
                }), 400


        input_data = pd.DataFrame(
            [data]
        )


        input_data["date"] = pd.to_datetime(
            input_data["date"]
        )


        input_data["date_year"] = (
            input_data["date"].dt.year
        )


        input_data["date_month"] = (
            input_data["date"].dt.month
        )


        input_data["date_day"] = (
            input_data["date"].dt.day
        )


        input_data.drop(
            "date",
            axis=1,
            inplace=True
        )


        city = str(
            input_data["city"].iloc[0]
        ).strip()


        country = str(
            input_data["country"].iloc[0]
        ).strip()


        if city not in city_map:

            return jsonify({
                "error":
                f"City '{city}' is not available in the training data."
            }), 400


        if country not in country_map:

            return jsonify({
                "error":
                f"Country '{country}' is not available in the training data."
            }), 400


        input_data["city"] = (
            input_data["city"].map(
                city_map
            )
        )


        input_data["country"] = (
            input_data["country"].map(
                country_map
            )
        )


        input_data = input_data[
            feature_columns
        ]


        prediction = model.predict(
            input_data
        )


        predicted_price = float(
            prediction[0]
        )


        return jsonify({

            "predicted_price":
            round(
                predicted_price,
                2
            )

        })


    except Exception as e:

        print(
            "Prediction Error:",
            e
        )

        return jsonify({

            "error":
            str(e)

        }), 500


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=True

    )