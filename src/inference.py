import pandas as pd

from config import (
RISK_THRESHOLD,
HIGH_RISK_THRESHOLD
)

class EAWSInference:

    def __init__(
        self,
        classifier,
        regressor,
        explainer,
        intervention_engine
    ):
        self.classifier = classifier
        self.regressor = regressor
        self.explainer = explainer
        self.intervention_engine = intervention_engine

    def prepare_input(self, student_data):
        """Convert student input into a DataFrame."""

        if isinstance(student_data, dict):
            student_data = pd.DataFrame([student_data])

        elif not isinstance(student_data, pd.DataFrame):
            raise TypeError(
                "Student data must be a dictionary or DataFrame."
            )

        return student_data.copy()

    def predict_risk(self, X):
        """Predict student risk probability."""

        risk_probability = self.classifier.predict_proba(
            X
        )[0, 1]

        if risk_probability >= HIGH_RISK_THRESHOLD:
            risk_level = "High"

        elif risk_probability >= RISK_THRESHOLD:
            risk_level = "At Risk"

        else:
            risk_level = "Low"

        return risk_probability, risk_level

    def predict_score(self, X):
        """Predict final exam score."""

        predicted_score = self.regressor.predict(X)[0]

        return predicted_score

    def assess_student(self, student_data):
        """Run complete EAWS assessment."""

        X = self.prepare_input(student_data)

        # Tier 1
        risk_probability, risk_level = (
            self.predict_risk(X)
        )

        # Tier 2
        predicted_score = self.predict_score(X)

        return {
            "risk_probability": risk_probability,
            "risk_level": risk_level,
            "predicted_score": predicted_score,
            "input_data": X
        }
