# main.py

from fastapi import FastAPI
from routes import router

app = FastAPI(
    title="Document Classifier",
    description="Upload document images to classify (Aadhar, PAN, etc.) or add new types.",
    version="1.0"
)

app.include_router(router, prefix="/doc")
