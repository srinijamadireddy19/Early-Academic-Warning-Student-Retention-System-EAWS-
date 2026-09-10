import pandas as pd

class FeatureEngineer:
"""
Creates predictor features and, separately, the classification target.

```
Predictor features can be created for both historical and new-student data.
The target 'at_risk' is created only from historical data during training.
"""

def __init__(self, data, engagement_stats=None):
    self.df = data.copy()
    self.engagement_stats = engagement_stats

def create_predictor_features(self):
    """
    Create features that are available at prediction time.

    These features must not depend on final_grade or final_exam_score.
    """
    self.create_attendance_risk()
    self.create_sleep_risk()
    self.create_workload_risk()
    self.create_engagement_score()

    return self.df

def create_target(self):
    """
    Create the classification target from historical final outcomes.

    at_risk:
        1 -> final_grade is D or F
        0 -> otherwise
    """
    if "final_grade" not in self.df.columns:
        raise ValueError(
            "final_grade is required to create the at_risk target."
        )

    self.df["at_risk"] = (
        self.df["final_grade"]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(["D", "F"])
        .astype(int)
    )

    return self.df["at_risk"]

def create_attendance_risk(self):
    """
    Create attendance risk category.
    """
    def attendance_risk(attendance):
        if attendance < 60:
            return "Critical"
        elif attendance < 75:
            return "Moderate"
        else:
            return "Healthy"

    if "attendance_percent" not in self.df.columns:
        raise ValueError(
            "attendance_percent column is required."
        )

    self.df["attendance_risk_level"] = (
        self.df["attendance_percent"].apply(attendance_risk)
    )

def create_sleep_risk(self):
    """
    Create sleep deviation features.
    """
    if "sleep_hours" not in self.df.columns:
        raise ValueError(
            "sleep_hours column is required."
        )

    self.df["sleep_deviation"] = (
        self.df["sleep_hours"] - 7.5
    ).abs()

    self.df["deviation_bin"] = self.df["sleep_deviation"].round()

def create_workload_risk(self):
    """
    Identify students who have a part-time job and sleep less than
    7 hours.
    """
    required_columns = ["part_time_job", "sleep_hours"]

    for column in required_columns:
        if column not in self.df.columns:
            raise ValueError(
                f"{column} column is required."
            )

    threshold = 7.0

    self.df["high_workload_low_sleep"] = (
        (
            self.df["part_time_job"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "yes"
        )
        & (self.df["sleep_hours"] < threshold)
    ).astype(int)

def create_engagement_score(self):
    """
    Create engagement score from:
    - attendance
    - study time
    - extracurricular activities

    Min/max values are learned from training data and reused for
    test/new-student data.
    """
    required_columns = [
        "attendance_percent",
        "study_time_hours",
        "extracurricular_activities"
    ]

    for column in required_columns:
        if column not in self.df.columns:
            raise ValueError(
                f"{column} column is required."
            )

    extracurricular_scaled = (
        self.df["extracurricular_activities"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1.0, "no": 0.0})
    )

    if extracurricular_scaled.isna().any():
        raise ValueError(
            "extracurricular_activities must contain only Yes/No values."
        )

    # Learn normalization statistics from this dataset during training.
    if self.engagement_stats is None:
        attendance_min = self.df["attendance_percent"].min()
        attendance_max = self.df["attendance_percent"].max()

        study_min = self.df["study_time_hours"].min()
        study_max = self.df["study_time_hours"].max()

        self.engagement_stats = {
            "attendance_min": attendance_min,
            "attendance_max": attendance_max,
            "study_min": study_min,
            "study_max": study_max
        }

    attendance_min = self.engagement_stats["attendance_min"]
    attendance_max = self.engagement_stats["attendance_max"]

    study_min = self.engagement_stats["study_min"]
    study_max = self.engagement_stats["study_max"]

    # Avoid division by zero if all values are identical.
    if attendance_max == attendance_min:
        attendance_scaled = pd.Series(
            0.0,
            index=self.df.index
        )
    else:
        attendance_scaled = (
            self.df["attendance_percent"] - attendance_min
        ) / (attendance_max - attendance_min)

    if study_max == study_min:
        study_scaled = pd.Series(
            0.0,
            index=self.df.index
        )
    else:
        study_scaled = (
            self.df["study_time_hours"] - study_min
        ) / (study_max - study_min)

    self.df["engagement_score"] = (
        attendance_scaled
        + study_scaled
        + extracurricular_scaled
    ) / 3

def get_engagement_stats(self):
    """
    Return the min/max statistics used to calculate engagement score.
    """
    return self.engagement_stats

def get_data(self):
    """
    Return the engineered dataframe.
    """
    return self.df
```
