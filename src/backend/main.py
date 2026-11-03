from fastapi import FastAPI
from src.backend.api.routes import router
app=FastAPI(title="Dr. ML Prediction App",version="1.0.0",description="Multiple Disease Prediction backend")
app.include_router(router,prefix="/api")
