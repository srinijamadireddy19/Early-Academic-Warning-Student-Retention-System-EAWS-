import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_path)
            print("Data loaded successfully.")
            return self.data
        except FileNotFoundError:
            print(f"File not found: {self.data_path}")

    def clean_data(self):
        df = self.data.copy()

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Handle missing values of parental_education by filling with unknown
        df['parental_education'].fillna('unknown', inplace=True)
        
        # clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        return df