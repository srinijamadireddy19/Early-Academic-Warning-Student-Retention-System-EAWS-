class Regressor:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = None

    def train_model(self):
        preprocessor_builder = DataPreprocessor()
        preprocessor = preprocessor_builder.build_preprocessor(self.X_train)

        self.model = Pipeline([
            ("preprocessor", preprocessor),
            ("lr", LinearRegression())
        ])
        print("Training the model...")
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed successfully.")
        return self.model

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        predictions = self.model.predict(X_test)
        return predictions

    def evaluate(self, y_test, predictions): 
        """Evaluate regression model.""" 
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
        linear_model = self.model.named_steps[ "model" ] 
        feature_names = ( preprocessor.get_feature_names_out() ) 
        coefficients = pd.DataFrame({ 
            "Feature": feature_names, 
            "Coefficient": linear_model.coef_ 
        }) 
        
        coefficients["Absolute_Coefficient"] = ( coefficients["Coefficient"].abs() ) 
        coefficients = coefficients.sort_values( by="Absolute_Coefficient", ascending=False ) 
        
        return coefficients

    def train_mapie(self, X_train, y_train):
    """Train MAPIE model for prediction intervals."""

    X_train_model, X_calib, y_train_model, y_calib = (
        train_test_split(
            X_train,
            y_train,
            test_size=0.25,
            random_state=42
        )
    )

    pipeline_builder = PreprocessingPipeline()

    preprocessor = pipeline_builder.build_preprocessor(
        X_train_model
    )

    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
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

    lower_bounds = intervals[:, 0, 0]
    upper_bounds = intervals[:, 1, 0]

    return predictions, lower_bounds, upper_bounds

}