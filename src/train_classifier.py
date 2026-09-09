from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data_preprocessing import DataPreprocessor
from src.config import RISK_THRESHOLD


class Classifier:
    def __init__(self):
        self.X_train_class = None
        self.y_train_class = None
        self.X_test_class = None
        self.y_test_class = None
        self.model = None

    def split_data(self, data, test_size=0.2, random_state=42):
        X = data.drop(columns=["final_grade","at_risk",'academic_level', 'final_exam_score'])
        y = data["at_risk"]

        (
            self.X_train_class,
            self.X_test_class,
            self.y_train_class,
            self.y_test_class,
        ) = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    def train_model(self):
        if self.X_train_class is None:
            raise ValueError("Data has not been split yet. Call split_data() first.")

        preprocessor_builder = DataPreprocessor()
        preprocessor = preprocessor_builder.build_preprocessor(
            self.X_train_class
        )

        positive_count = (self.y_train_class == 1).sum()
        negative_count = (self.y_train_class == 0).sum()

        if positive_count == 0:
            raise ValueError("No positive class samples found in training data.")

        scale_pos_weight = negative_count / positive_count

        self.model = Pipeline([
            ("preprocessor", preprocessor),
            ("xgb", XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                scale_pos_weight=scale_pos_weight,
            ))
        ])

        print("Training the model...")
        self.model.fit(self.X_train_class, self.y_train_class)
        print("Model training completed successfully.")

        return self.model

    def predict(self, X_test=None):
        if self.model is None:
            raise ValueError(
                "Model has not been trained yet. Call train_model() first."
            )

        if X_test is None:
            X_test = self.X_test_class

        probabilities = self.model.predict_proba(X_test)[:, 1]

        predictions = (
            probabilities >= RISK_THRESHOLD
        ).astype(int)

        return predictions, probabilities

    def evaluate(self, y_test=None, predictions=None):
        """Evaluate classification model."""

        if y_test is None:
            y_test = self.y_test_class

        if predictions is None:
            predictions, _ = self.predict(self.X_test_class)

        probabilities = self.model.predict_proba(self.X_test_class)[:, 1]

        results = {
            "Recall": recall_score(y_test, predictions,average='weighted'),
            "Precision": precision_score(y_test, predictions,average='weighted'),
            "F1 Score": f1_score(y_test, predictions,average='weighted'),
            "ROC AUC": roc_auc_score(y_test, probabilities),
        }

        return results