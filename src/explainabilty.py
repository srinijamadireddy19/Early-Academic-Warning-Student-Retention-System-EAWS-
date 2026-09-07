import pandas as pd
import shap

class Explainer:

```
def __init__(self, classification_model, regression_model):
    self.classification_model = classification_model
    self.regression_model = regression_model

def explain_classifier(self, X):
    """Generate SHAP values for classification model."""

    preprocessor = self.classification_model.named_steps["preprocessor"]
    model = self.classification_model.named_steps["xgb"]

    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_transformed)

    return shap_values, feature_names

def explain_regressor(self, X):
    """Generate SHAP values for regression model."""

    preprocessor = self.regression_model.named_steps["preprocessor"]
    model = self.regression_model.named_steps["model"]

    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.LinearExplainer(
        model,
        X_transformed
    )

    shap_values = explainer.shap_values(
        X_transformed
    )

    return shap_values, feature_names

def get_original_feature_name(self, feature_name):
    """Convert transformed feature name to original feature name."""

    name = feature_name.replace("categorical__", "")
    name = name.replace("numeric__", "")
    name = name.replace("cat__", "")
    name = name.replace("num__", "")

    categorical_features = [
        "gender",
        "parental_education",
        "internet_access",
        "part_time_job",
        "academic_level",
        "attendance_risk_level"
    ]

    for feature in categorical_features:
        if name.startswith(feature + "_"):
            return feature

    return name

def get_top_drivers(
    self,
    shap_values,
    feature_names,
    student_data,
    top_n=5
):
    """Get most important factors for one student."""

    drivers = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values
    })

    # Convert encoded features to original features
    drivers["original_feature"] = (
        drivers["feature"]
        .apply(self.get_original_feature_name)
    )

    # Combine SHAP values from encoded categories
    drivers = (
        drivers.groupby(
            "original_feature",
            as_index=False
        )["shap_value"]
        .sum()
    )

    # Calculate impact
    drivers["abs_impact"] = (
        drivers["shap_value"].abs()
    )

    # Add student's actual feature value
    student = student_data.iloc[0]

    drivers["student_value"] = (
        drivers["original_feature"].map(student)
    )

    # Sort by importance
    drivers = drivers.sort_values(
        "abs_impact",
        ascending=False
    ).head(top_n)

    return drivers

def add_contribution_direction(
    self,
    drivers,
    model_type
):
    """Add explanation for SHAP direction."""

    drivers = drivers.copy()

    if model_type == "classification":

        drivers["contribution"] = (
            drivers["shap_value"]
            .apply(
                lambda x:
                "Increases Risk"
                if x > 0
                else "Reduces Risk"
            )
        )

    elif model_type == "regression":

        drivers["contribution"] = (
            drivers["shap_value"]
            .apply(
                lambda x:
                "Increases Predicted Score"
                if x > 0
                else "Decreases Predicted Score"
            )
        )

    return drivers
```
