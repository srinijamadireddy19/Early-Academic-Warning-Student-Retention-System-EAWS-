import pandas as pd
import numpy as np


class DataPreprocessor:
    def build_preprocessor(self, data):
        numeric_features = data.select_dtypes( include=["number"] ).columns 
        # Identify categorical columns 
        categorical_features = data.select_dtypes( include=["object", "category", "bool"] ).columns 
        # Numerical preprocessing 
        numeric_pipeline = Pipeline([ 
            ("imputer", SimpleImputer(strategy="median")), 
            ("scaler", StandardScaler()) 
        ])
            
         # Categorical preprocessing 
         categorical_pipeline = Pipeline([ 
            ("imputer", SimpleImputer(strategy="most_frequent")),
             ("encoder", OneHotEncoder(handle_unknown="ignore")) 
            ]) 
            
        # Combine both pipelines 
        preprocessor = ColumnTransformer([ 
            ("numeric", numeric_pipeline, numeric_features), 
            ("categorical", categorical_pipeline, categorical_features) 
        ]) 
        
        return preprocessor

