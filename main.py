from src.data_cleaner import DataCleaner
from src.feature_engineer import FeatureEngineer
from src.data_preprocessing import DataPreprocessor
from src.train_classifier import Classifier
from src.train_regressor import Regressor

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH
)


def main():
    print("Loading data...")

    cleaner = DataCleaner(RAW_DATA_PATH)
    df = cleaner.load_data()
    print("Data loaded successfully.")

    df = cleaner.clean_data()
    print("Data cleaned successfully.")

    print("Performing feature engineering...")
    engineer = FeatureEngineer(df)
    engineer.create_features()
    print("Feature engineering completed successfully.")

    print("Saving processed data...")
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}.")

    classifier = Classifier()
    classifier.split_data(df)
    classifier.train_model()

    results = classifier.evaluate()

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")


    print("Regression model training and evaluation.")
    regressor = Regressor()
    regressor.split_data(df)
    regressor.train_model()

    results_reg = regressor.evaluate()

    for metric, value in results_reg.items():
        print(f"{metric}: {value:.4f}")

    coefficients = regressor.get_feature_coefficients()
    print(coefficients)

    print("\nMAPIE prediction intervals:")

    regressor.train_mapie(df)

    predictions, lower_bounds, upper_bounds = (
        regressor.predict_with_intervals(regressor.X_test_class)
    )

    for i in range(min(10, len(predictions))):
        print(
            f"Prediction: {predictions[i]:.2f}, "
            f"Lower: {lower_bounds[i]:.2f}, "
            f"Upper: {upper_bounds[i]:.2f}"
        )


    



main()