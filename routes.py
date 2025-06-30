# routes.py

from fastapi import APIRouter, UploadFile, File, Form
from typing import List
from PIL import Image
import io

from doc import add_class_from_images, classify_image_pil

router = APIRouter()


@router.post("/add-class/")
async def add_class_route(
    class_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    images = []
    for file in files:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        images.append(image)

    success = add_class_from_images(class_name, images)
    return {"status": "success" if success else "failure"}


@router.post("/predict/")
async def classify_route(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    result = classify_image_pil(image)
    return result

