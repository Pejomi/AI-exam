import joblib
from keras.models import load_model

model_path = "../models/accident_severity_model.keras"
preprocessor_path = "../models/preprocessor.joblib"

def save_model_and_preprocessor(model, preprocessor):
    # Save the trained model
    model.save(model_path)
    # Save the preprocessor
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {preprocessor_path}")

def load_model_and_preprocessor():
    # Load the trained model
    model = load_model(model_path)
    # Load the preprocessor
    preprocessor = joblib.load(preprocessor_path)
    print(f"Model loaded from {model_path}")
    print(f"Preprocessor loaded from {preprocessor_path}")
    return model, preprocessor
