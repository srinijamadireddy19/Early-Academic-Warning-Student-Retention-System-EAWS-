import joblib 

def save_model(model, model_path):
    joblib.dump(model, model_path)

def load_model(model_path):
    return joblib.load(model_path)

def ensure_directory_exists(directory_path):
    import os
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

