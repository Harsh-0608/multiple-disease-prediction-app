from typing import Dict
from pydantic import BaseModel
class PredictionRequest(BaseModel):
    disease:str
    features:Dict[str,int|float|str]
class PredictionResponse(BaseModel):
    disease:str
    prediction:int
    probability:float