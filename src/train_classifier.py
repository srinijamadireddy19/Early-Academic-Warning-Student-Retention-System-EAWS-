from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder



class Classifier:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = None

    def train_model(self):
        preprocessor_builder = DataPreprocessor()
        preprocessor = preprocessor_builder.build_preprocessor(self.X_train)

        scale_pos_weight = ( (self.y_train == 0).sum() / (self.y_train == 1).sum() )
        
        self.model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=5,
                random_state=42, 
                scale_pos_weight=scale_pos_weight
            ))
        ])
        print("Training the model...")
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed successfully.")
        return self.model


    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        # Get probability of positive class 
        probabilities = self.model.predict_proba(X_test)[:, 1] 
        # Apply threshold 
        predictions = ( probabilities >= self.threshold ).astype(int) 
        return predictions, probabilities

    def evaluate(self, y_test, predictions): 
        """Evaluate classification model."""
        results = { 
            "Recall": recall_score(y_test, predictions), 
            "Precision": precision_score(y_test, predictions), 
            "F1 Score": f1_score(y_test, predictions) 
        } 
            
        return results

    
