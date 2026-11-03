from fastapi import APIRouter
from src.backend.schemas.prediction_schema import(
    PredictionRequest,
    PredictionResponse
)
from src.backend.services.predictor import predict_disease
router=APIRouter()
@router.get("/health")
def health_check():
    return {"status":"ok"}
@router.post("/predict",response_model=PredictionResponse)
def predict(request:PredictionRequest):
    disease=request.disease
    input_data=request.features
    prediction_result=predict_disease(disease=disease,input_data=input_data)
    return PredictionResponse(**prediction_result)
