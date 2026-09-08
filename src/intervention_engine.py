class InterventionEngine:

```
ACTIONABLE_FEATURES = [
    "attendance_percent",
    "attendance_risk_level",
    "study_time_hours",
    "engagement_score",
    "previous_grade",
    "part_time_job",
    "internet_access"
]

SEVERITY_THRESHOLDS = {

    "attendance_percent": {
        "high": 60,
        "medium": 75
    },

    "study_time_hours": {
        "high": 2,
        "medium": 4
    },

    "engagement_score": {
        "high": 0.3,
        "medium": 0.5
    },

    "previous_grade": {
        "high": 50,
        "medium": 65
    }
}

INTERVENTION_RULES = {

    "attendance_percent": {
        "high": {
            "recommendation": "Immediate Academic Advisor Check-In",
            "reason": "Low attendance is strongly associated with academic risk."
        },
        "medium": {
            "recommendation": "Attendance Improvement Plan",
            "reason": "Attendance patterns may negatively affect academic progress."
        },
        "low": {
            "recommendation": "Routine Attendance Monitoring",
            "reason": "Continue monitoring attendance patterns."
        }
    },
        "attendance_risk_level": {
        "high": {
            "recommendation": "Immediate Attendance Intervention",
            "reason": "The student's attendance risk level requires immediate attention."
        },
        "moderate": {
            "recommendation": "Attendance Improvement Plan",
            "reason": "Attendance requires targeted monitoring and support."
        },
        "healthy": {
            "recommendation": "No Attendance Intervention Needed",
            "reason": "Current attendance pattern appears healthy."
        }
    },
        "study_time_hours": {
        "high": {
            "recommendation": "Personalized Study Plan",
            "reason": "Study time may be insufficient for academic requirements."
        },
        "medium": {
            "recommendation": "Time Management and Study Skills Support",
            "reason": "Improving study routines may support academic performance."
        },
        "low": {
            "recommendation": "Maintain Current Study Routine",
            "reason": "Current study habits appear supportive."
        }
    },
        "engagement_score": {
        "high": {
            "recommendation": "Student Engagement and Mentoring Program",
            "reason": "Low engagement may indicate reduced academic participation."
        },
        "medium": {
            "recommendation": "Faculty or Peer Mentoring Check-In",
            "reason": "Additional engagement support may improve participation."
        },
        "low": {
            "recommendation": "Routine Engagement Monitoring",
            "reason": "Continue monitoring student participation."
        }
    },
        "previous_grade": {
        "high": {
            "recommendation": "Academic Tutoring and Recovery Plan",
            "reason": "Previous academic performance suggests a need for intensive support."
        },
        "medium": {
            "recommendation": "Subject-Specific Academic Support",
            "reason": "Additional academic guidance may improve future performance."
        },
        "low": {
            "recommendation": "Maintain Academic Progress",
            "reason": "Previous academic performance is satisfactory."
        }
    },
        "part_time_job": {
        "high": {
            "recommendation": "Academic Workload and Financial Support Consultation",
            "reason": "External work commitments may be affecting academic balance."
        },
        "medium": {
            "recommendation": "Workload Management Consultation",
            "reason": "Review balance between employment and academic responsibilities."
        },
        "low": {
            "recommendation": "No Intervention Based on Employment Status Alone",
            "reason": "Employment status alone should not trigger intervention."
        }
    },
        "internet_access": {
        "high": {
            "recommendation": "Technology and Digital Access Support",
            "reason": "Limited internet access may restrict access to academic resources."
        },
        "medium": {
            "recommendation": "Campus Digital Resource Support",
            "reason": "Ensure the student has reliable access to required learning resources."
        },
        "low": {
            "recommendation": "No Digital Access Intervention Needed",
            "reason": "Current internet access appears adequate."
        }
    }
}

def get_feature_severity(self, feature, value):
    """Determine severity of a student's feature."""

    thresholds = self.SEVERITY_THRESHOLDS.get(feature)

    # Numerical features
    if thresholds:

        if value < thresholds["high"]:
            return "high"

        elif value < thresholds["medium"]:
            return "medium"

        return "low"

    # Part-time job
    if feature == "part_time_job":

        if str(value).lower() == "yes":
            return "medium"

        return "low"

    # Internet access
    if feature == "internet_access":

        if str(value).lower() == "no":
            return "high"

        return "low"

    # Attendance risk
    if feature == "attendance_risk_level":

        value = str(value).lower()

        if value in ["critical", "poor", "high"]:
            return "high"

        elif value in ["moderate", "medium"]:
            return "medium"

        return "low"

    return None

def identify_needs(
    self,
    classification_drivers,
    regression_drivers
):
    """Identify actionable intervention needs."""

    needs = []

    # Classification model
    for _, row in classification_drivers.iterrows():

        feature = row["original_feature"]

        if feature not in self.ACTIONABLE_FEATURES:
            continue

        severity = self.get_feature_severity(
            feature,
            row["student_value"]
        )

        # Positive SHAP increases risk
        if severity in ["high", "medium"] and row["shap_value"] > 0:

            needs.append({
                "feature": feature,
                "value": row["student_value"],
                "severity": severity,
                "source": "risk_model"
            })

    # Regression model
    for _, row in regression_drivers.iterrows():

        feature = row["original_feature"]

        if feature not in self.ACTIONABLE_FEATURES:
            continue

        severity = self.get_feature_severity(
            feature,
            row["student_value"]
        )

        # Negative SHAP decreases score
        if severity in ["high", "medium"] and row["shap_value"] < 0:

            needs.append({
                "feature": feature,
                "value": row["student_value"],
                "severity": severity,
                "source": "performance_model"
            })

    return needs

def generate_interventions(self, needs):
    """Generate recommendations from intervention needs."""

    recommendations = []

    seen = set()

    for need in needs:

        feature = need["feature"]
        severity = need["severity"]

        rule = self.INTERVENTION_RULES.get(
            feature,
            {}
        )

        recommendation = rule[severity]["recommendation"]
        reason = rule[severity]["reason"]

        if recommendation and recommendation not in seen:

            recommendations.append({
                "feature": feature,
                "severity": severity,
                "recommendation": recommendation
            })

            seen.add(recommendation)

    return recommendations

def get_priority(
    self,
    risk_probability,
    predicted_score
):
    """Determine overall intervention priority."""

    if risk_probability >= 0.70 or predicted_score < 50:
        return "HIGH"

    elif risk_probability >= 0.30 or predicted_score < 65:
        return "MEDIUM"

    return "LOW"

def get_final_recommendations(
    self,
    risk_probability,
    predicted_score,
    recommendations
):
    """Generate final intervention decision."""

    priority = self.get_priority(
        risk_probability,
        predicted_score
    )

    # Low priority
    if priority == "LOW":

        return {
            "priority": "LOW",
            "action": "Routine Monitoring",
            "recommendations": [
                "Continue Routine Academic Monitoring"
            ]
        }

    # No specific intervention identified
    if not recommendations:

        return {
            "priority": priority,
            "action": "Advisor Review Recommended",
            "recommendations": [
                "Academic Advisor Review"
            ]
        }

    # Targeted intervention
    return {
        "priority": priority,
        "action": "Targeted Intervention",
        "recommendations": recommendations
    }

def recommend(
    self,
    classification_drivers,
    regression_drivers,
    risk_probability,
    predicted_score
):
    """Run complete intervention pipeline."""

    needs = self.identify_needs(
        classification_drivers,
        regression_drivers
    )

    recommendations = self.generate_interventions(
        needs
    )

    final_result = self.get_final_recommendations(
        risk_probability,
        predicted_score,
        recommendations
    )

    return final_result
```
