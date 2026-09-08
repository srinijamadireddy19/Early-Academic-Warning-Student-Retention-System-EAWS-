# Paths
RAW_DATA_PATH = "data/raw/student_performance_dataset.csv"
PROCESSED_DATA_PATH = "data/processed/student_performance_dataset_cleaned.csv"
CLASSIFIER_MODEL_PATH = "models/classifier.pkl"
REGRESSOR_MODEL_PATH = "models/regressor.pkl"

# Targets
CLASSIFICATION_TARGET = "at_risk"
REGRESSION_TARGET = "final_exam_score"

# Business thresholds
RISK_THRESHOLD = 0.15
HIGH_RISK_THRESHOLD = 0.70

# Random state
RANDOM_STATE = 42
TEST_SIZE = 0.2