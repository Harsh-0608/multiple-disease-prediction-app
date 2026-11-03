import logging
from pathlib import Path
import pandas as pd
from joblib import load
from src.backend.config.settings import Settings
import __main__

from src.training.diabetes import replace_zeros_with_nan

__main__.replace_zeros_with_nan = replace_zeros_with_nan
settings=Settings()
DIABETES_MODEL_PATH=Path(settings.diabetes_model_path)
HEART_DISEASE_MODEL_PATH=Path(settings.heart_disease_model_path)
LOG_PATH=Path(settings.LOG_PATH)
LOG_PATH.parent.mkdir(parents=True,exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logging.info("Loading models for prediction")
logging.info(f"Diabetes model path: {DIABETES_MODEL_PATH}")
logging.info(f"Diabetes model exists: {DIABETES_MODEL_PATH.exists()}")
logging.info(f"Diabetes model size: {DIABETES_MODEL_PATH.stat().st_size}")
diabetes_model=load(DIABETES_MODEL_PATH)
heart_disease_model=load(HEART_DISEASE_MODEL_PATH)
logging.info("Models loaded successfully for prediction")
def predict_disease(disease:str,input_data:dict):
    if(disease=='diabetes'):
        model=diabetes_model
    elif(disease=='heart_disease'):
        model=heart_disease_model
    else:
        raise ValueError("Invalid disease type. Choose 'diabetes' or 'heart_disease'.")
    input_df=pd.DataFrame([input_data])
    prediction=int(model.predict(input_df)[0])
    probability=float(model.predict_proba(input_df)[0][1])
    logging.info(
        f"[{disease}] prediction={prediction}, probability={probability}"
    )

    return {
        "disease": disease,
        "prediction": prediction,
        "probability": probability,
    }


# Example usage - comment out in production
# diabetes_input = {
#     "Pregnancies": 2,
#     "Glucose": 120,
#     "BloodPressure": 70,
#     "SkinThickness": 25,
#     "Insulin": 80,
#     "BMI": 28.5,
#     "DiabetesPedigreeFunction": 0.5,
#     "Age": 30
# }

# heart_input = {
#     "age": 52,
#     "sex": 1,
#     "cp": 0,
#     "trestbps": 125,
#     "chol": 212,
#     "fbs": 0,
#     "restecg": 1,
#     "thalach": 168,
#     "exang": 0,
#     "oldpeak": 1.0,
#     "slope": 2,
#     "ca": 0,
#     "thal": 2
# }


# print(predict_disease("diabetes", diabetes_input))
# print(predict_disease("heart_disease", heart_input))
