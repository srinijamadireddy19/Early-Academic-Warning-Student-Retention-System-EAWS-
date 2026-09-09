import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mapie.regression import SplitConformalRegressor

from src.data_preprocessing import DataPreprocessor

class Regressor:
    def __init__(self):
        self.X_train_class = None
        self.y_train_class = None
        self.X_test_class = None
        self.y_test_class = None
        self.model = None

    def split_data(self, data, test_size=0.2, random_state=42):
        X = data.drop(columns=["student_id","final_grade","at_risk",'academic_level', 'final_exam_score'])
        y = data["final_exam_score"]

        (
           self.X_train_class,
            self.X_test_class,
            self.y_train_class,
            self.y_test_class,
        ) = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
            )

    def train_model(self):

        if self.X_train_class is None:
            raise ValueError("Data has not been split yet. Call split_data() first.")

        preprocessor_builder = DataPreprocessor()
        preprocessor = preprocessor_builder.build_preprocessor(
            self.X_train_class
        )

        self.model = Pipeline([
            ("preprocessor", preprocessor),
            ("lr", LinearRegression())
        ])
        print("Training the model...")
        self.model.fit(self.X_train_class, self.y_train_class)
        print("Model training completed successfully.")
        return self.model

    def predict(self, X_test=None):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        if X_test is None:
            X_test = self.X_test_class
        predictions = self.model.predict(X_test)
        return predictions

    def evaluate(self, y_test=None, predictions=None): 
        """Evaluate regression model.""" 
        if y_test is None:
            y_test = self.y_test_class
        if predictions is None:
            predictions = self.predict()

        results = { 
            "MAE": mean_absolute_error( y_test, predictions ), 
            "MSE": mean_squared_error( y_test, predictions ),
             "RMSE": mean_squared_error( y_test, predictions ) ** 0.5,
            "R2 Score": r2_score( y_test, predictions ) 
        } 
        
        return results

    def get_feature_coefficients(self): 
        """Get feature importance using Linear Regression coefficients.""" 
        preprocessor = self.model.named_steps[ "preprocessor" ] 
        linear_model = self.model.named_steps[ "lr" ] 
        feature_names = ( preprocessor.get_feature_names_out() ) 
        coefficients = pd.DataFrame({ 
            "Feature": feature_names, 
            "Coefficient": linear_model.coef_ 
        }) 
        
        coefficients["Absolute_Coefficient"] = ( coefficients["Coefficient"].abs() ) 
        coefficients = coefficients.sort_values( by="Absolute_Coefficient", ascending=False ) 
        
        return coefficients

    def train_mapie(self, data):
        """Train MAPIE model for prediction intervals."""

        X = data.drop(columns=["student_id","final_grade","at_risk",'academic_level', 'final_exam_score'])
        y = data["final_exam_score"]

        X_train_model, X_calib, y_train_model, y_calib = (
            train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=42
            )
        )

        pipeline_builder = DataPreprocessor()

        preprocessor = pipeline_builder.build_preprocessor(
            X_train_model
        )

        lr_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("lr", LinearRegression())
        ])

        self.mapie_model = SplitConformalRegressor(
            estimator=lr_pipeline,
            confidence_level=0.95,
            prefit=False
        )

        # Train model
        self.mapie_model.fit(
            X_train_model,
            y_train_model
        )

        # Calibrate prediction intervals
        self.mapie_model.conformalize(
            X_calib,
            y_calib
        )

        return self.mapie_model

    def predict_with_intervals(self, X_test):
        """Predict with 95% prediction intervals."""

        predictions, intervals = (
            self.mapie_model.predict_interval(X_test)
        )

        lower_bounds = np.clip(intervals[:, 0, 0], 0, 100)
        upper_bounds = np.clip(intervals[:, 1, 0], 0, 100)
        predictions = np.clip(predictions, 0, 100)
        
        return predictions, lower_bounds, upper_bounds

    