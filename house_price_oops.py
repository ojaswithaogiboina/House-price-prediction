import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


class HousePricePrediction:

    # ============================================================
    # 1. CONSTRUCTOR
    # ============================================================
    def __init__(self, file_name):

        try:
            self.file_name = file_name

            self.model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )

            self.city_map = {}
            self.country_map = {}
            self.feature_columns = []

            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None

            print("Constructor created successfully.")

        except Exception as e:
            print("Error in constructor:", e)

    # ============================================================
    # 2. DATA PREPARATION
    # ============================================================
    def prepare_data(self):

        try:

            # Read dataset
            df = pd.read_csv(self.file_name)

            print("\n========================================")
            print("           DATA INFORMATION")
            print("========================================")

            print("Dataset Shape:", df.shape)

            # Check null values
            null_values = df.isnull().sum().sum()

            print("Null Values:", null_values)

            # ----------------------------------------------------
            # DATE PROCESSING
            # ----------------------------------------------------

            df["date"] = pd.to_datetime(
                df["date"],
                format="mixed"
            )

            # Extract date information
            df["date_year"] = df["date"].dt.year
            df["date_month"] = df["date"].dt.month
            df["date_day"] = df["date"].dt.day

            # Remove original date column
            df.drop(
                "date",
                axis=1,
                inplace=True
            )

            # ----------------------------------------------------
            # CITY MAPPING
            # ----------------------------------------------------

            cities = sorted(
                df["city"].dropna().unique()
            )

            self.city_map = {
                city: number
                for number, city in enumerate(cities)
            }

            df["city"] = df["city"].map(
                self.city_map
            )

            # ----------------------------------------------------
            # COUNTRY MAPPING
            # ----------------------------------------------------

            countries = sorted(
                df["country"].dropna().unique()
            )

            self.country_map = {
                country: number
                for number, country in enumerate(countries)
            }

            df["country"] = df["country"].map(
                self.country_map
            )

            # ----------------------------------------------------
            # REMOVE NULL VALUES
            # ----------------------------------------------------

            df.dropna(inplace=True)

            print("Null Values After Processing:",
                  df.isnull().sum().sum())

            # ----------------------------------------------------
            # X AND Y
            # ----------------------------------------------------

            # Price is dependent variable
            y = df["price"]

            # Remaining columns are independent variables
            X = df.drop(
                "price",
                axis=1
            )

            # Store feature columns
            self.feature_columns = X.columns.tolist()

            # ----------------------------------------------------
            # TRAIN TEST SPLIT
            # ----------------------------------------------------

            (
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test
            ) = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42
            )

            print("\n========================================")
            print("          DATA SPLIT")
            print("========================================")

            print(
                "Training Data:",
                self.X_train.shape
            )

            print(
                "Testing Data :",
                self.X_test.shape
            )

            print("\nCity Mapping:")
            print(self.city_map)

            print("\nCountry Mapping:")
            print(self.country_map)

            return True

        except Exception as e:

            print(
                "\nError in data preparation:",
                e
            )

            return False

    # ============================================================
    # 3. TRAINING
    # ============================================================
    def train_model(self):

        try:

            print("\n========================================")
            print("             MODEL TRAINING")
            print("========================================")

            self.model.fit(
                self.X_train,
                self.y_train
            )

            # Prediction on training data
            train_prediction = self.model.predict(
                self.X_train
            )

            # ----------------------------------------------------
            # TRAIN MSE - MANUAL
            # ----------------------------------------------------

            train_mse = np.mean(
                (
                    self.y_train.values
                    - train_prediction
                ) ** 2
            )

            # ----------------------------------------------------
            # TRAIN RMSE - MANUAL
            # ----------------------------------------------------

            train_rmse = np.sqrt(
                train_mse
            )

            # ----------------------------------------------------
            # TRAIN R2 - MANUAL
            # ----------------------------------------------------

            numerator = np.sum(
                (
                    self.y_train.values
                    - train_prediction
                ) ** 2
            )

            denominator = np.sum(
                (
                    self.y_train.values
                    - self.y_train.mean()
                ) ** 2
            )

            train_r2 = 1 - (
                numerator / denominator
            )

            # ----------------------------------------------------
            # TRAIN ACCURACY
            # Within 10% of actual price
            # ----------------------------------------------------

            train_accuracy = np.mean(
                np.abs(
                    (
                        self.y_train.values
                        - train_prediction
                    )
                    / self.y_train.values
                ) <= 0.10
            ) * 100

            print("\n========== TRAIN RESULTS ==========")

            print(
                f"Train Accuracy : "
                f"{train_accuracy:.2f}%"
            )

            print(
                f"Train Loss (MSE): "
                f"{train_mse:.2f}"
            )

            print(
                f"Train RMSE      : "
                f"{train_rmse:.2f}"
            )

            print(
                f"Train R2        : "
                f"{train_r2:.4f}"
            )

            return True

        except Exception as e:

            print(
                "\nError during training:",
                e
            )

            return False

    # ============================================================
    # 4. TESTING
    # ============================================================
    def test_model(self):

        try:

            # Prediction on testing data
            test_prediction = self.model.predict(
                self.X_test
            )

            # ----------------------------------------------------
            # TEST MSE - MANUAL
            # ----------------------------------------------------

            test_mse = np.mean(
                (
                    self.y_test.values
                    - test_prediction
                ) ** 2
            )

            # ----------------------------------------------------
            # TEST RMSE - MANUAL
            # ----------------------------------------------------

            test_rmse = np.sqrt(
                test_mse
            )

            # ----------------------------------------------------
            # TEST R2 - MANUAL
            # ----------------------------------------------------

            numerator = np.sum(
                (
                    self.y_test.values
                    - test_prediction
                ) ** 2
            )

            denominator = np.sum(
                (
                    self.y_test.values
                    - self.y_test.mean()
                ) ** 2
            )

            test_r2 = 1 - (
                numerator / denominator
            )

            # ----------------------------------------------------
            # TEST ACCURACY
            # Within 10% of actual price
            # ----------------------------------------------------

            test_accuracy = np.mean(
                np.abs(
                    (
                        self.y_test.values
                        - test_prediction
                    )
                    / self.y_test.values
                ) <= 0.10
            ) * 100

            print("\n========== TEST RESULTS ==========")

            print(
                f"Test Accuracy  : "
                f"{test_accuracy:.2f}%"
            )

            print(
                f"Test Loss (MSE): "
                f"{test_mse:.2f}"
            )

            print(
                f"Test RMSE      : "
                f"{test_rmse:.2f}"
            )

            print(
                f"Test R2        : "
                f"{test_r2:.4f}"
            )

            return True

        except Exception as e:

            print(
                "\nError during testing:",
                e
            )

            return False

    # ============================================================
    # 5. SAVE MODEL
    # ============================================================
    def save_model(self):

        try:

            model_data = {

                "model": self.model,

                "city_map": self.city_map,

                "country_map": self.country_map,

                "feature_columns":
                    self.feature_columns
            }

            joblib.dump(
                model_data,
                "house_price_model.pkl"
            )

            print("\n========================================")
            print("           MODEL SAVING")
            print("========================================")

            print(
                "Model saved successfully!"
            )

            print(
                "File: house_price_model.pkl"
            )

            return True

        except Exception as e:

            print(
                "\nError while saving model:",
                e
            )

            return False

    # ============================================================
    # 6. LOAD MODEL AND PREDICT TEST POINTS
    # ============================================================
    def load_model_and_predict(self):

        try:

            print("\n========================================")
            print("       MODEL LOADING & PREDICTION")
            print("========================================")

            # Load saved model
            saved_data = joblib.load(
                "house_price_model.pkl"
            )

            loaded_model = saved_data["model"]

            loaded_city_map = \
                saved_data["city_map"]

            loaded_country_map = \
                saved_data["country_map"]

            loaded_features = \
                saved_data["feature_columns"]

            print(
                "Model loaded successfully!"
            )

            # ====================================================
            # TEST POINT 1
            # ====================================================

            test_point_1 = {

                "date": "2014-05-02",

                "bedrooms": 3,

                "bathrooms": 2.0,

                "sqft_living": 1800,

                "sqft_lot": 8000,

                "floors": 1.0,

                "waterfront": 0,

                "view": 0,

                "condition": 4,

                "sqft_above": 1800,

                "sqft_basement": 0,

                "yr_built": 1970,

                "yr_renovated": 0,

                "city": "Seattle",

                "country": "USA"
            }

            # ====================================================
            # TEST POINT 2
            # ====================================================

            test_point_2 = {

                "date": "2014-06-15",

                "bedrooms": 4,

                "bathrooms": 2.5,

                "sqft_living": 2500,

                "sqft_lot": 9000,

                "floors": 2.0,

                "waterfront": 0,

                "view": 1,

                "condition": 4,

                "sqft_above": 2200,

                "sqft_basement": 300,

                "yr_built": 1990,

                "yr_renovated": 0,

                "city": "Bellevue",

                "country": "USA"
            }

            test_points = [
                test_point_1,
                test_point_2
            ]

            # ====================================================
            # PREDICT TEST POINTS
            # ====================================================

            for number, point in enumerate(
                test_points,
                start=1
            ):

                point_df = pd.DataFrame(
                    [point]
                )

                # ------------------------------------------------
                # DATE
                # ------------------------------------------------

                point_df["date"] = \
                    pd.to_datetime(
                        point_df["date"],
                        format="mixed"
                    )

                point_df["date_year"] = \
                    point_df["date"].dt.year

                point_df["date_month"] = \
                    point_df["date"].dt.month

                point_df["date_day"] = \
                    point_df["date"].dt.day

                point_df.drop(
                    "date",
                    axis=1,
                    inplace=True
                )

                # ------------------------------------------------
                # CITY MAP
                # ------------------------------------------------

                point_df["city"] = \
                    point_df["city"].map(
                        loaded_city_map
                    )

                # ------------------------------------------------
                # COUNTRY MAP
                # ------------------------------------------------

                point_df["country"] = \
                    point_df["country"].map(
                        loaded_country_map
                    )

                # ------------------------------------------------
                # COLUMN ORDER
                # ------------------------------------------------

                point_df = point_df[
                    loaded_features
                ]

                # ------------------------------------------------
                # PREDICTION
                # ------------------------------------------------

                prediction = \
                    loaded_model.predict(
                        point_df
                    )

                print(
                    f"\nTest Point {number}:"
                )

                print(
                    f"Predicted Price: "
                    f"${prediction[0]:,.2f}"
                )

            return True

        except Exception as e:

            print(
                "\nError while loading model "
                "or predicting:",
                e
            )

            return False


# ================================================================
# 7. MAIN METHOD
# ================================================================
def main():

    try:

        print("\n")
        print("========================================")
        print("     HOUSE PRICE PREDICTION PROJECT")
        print("========================================")

        # Create object
        predictor = HousePricePrediction(
            "data.csv"
        )

        # Prepare data
        data_status = \
            predictor.prepare_data()

        if data_status:

            # Train model
            train_status = \
                predictor.train_model()

            if train_status:

                # Test model
                test_status = \
                    predictor.test_model()

                if test_status:

                    # Save model
                    save_status = \
                        predictor.save_model()

                    if save_status:

                        # Load model
                        predictor.load_model_and_predict()

        print("\n========================================")
        print("             PROJECT COMPLETED")
        print("========================================")

    except Exception as e:

        print(
            "\nError in main:",
            e
        )


# ================================================================
# PROGRAM START
# ================================================================
if __name__ == "__main__":
    main()