from src.data_cleaner import DataCleaner
from src.feature_engineer import FeatureEngineer
from src.data_preprocessing import DataPreprocessor
from src.train_classifier import Classifier

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


    



main()