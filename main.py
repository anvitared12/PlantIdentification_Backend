from dotenv import load_dotenv
load_dotenv()

import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from model_loader import load_keras_model, predict_plant
from plantnet import query_plantnet

app = FastAPI(title="Plant Identification API")

MODEL_PATH = os.getenv("MODEL_PATH", "plant_model.tflite")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "class_names.txt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))

model, class_names, img_size = load_keras_model(MODEL_PATH, CLASS_NAMES_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/identify")
async def identify_plant(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10 MB).")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    keras_result = predict_plant(model, class_names, image, img_size)

    if keras_result["confidence"] >= CONFIDENCE_THRESHOLD:
        return JSONResponse({
            "source": "local_model",
            "plant_name": keras_result["plant_name"],
            "confidence": round(keras_result["confidence"], 4),
        })

    plantnet_result = await query_plantnet(raw, file.filename or "image.jpg")

    if plantnet_result:
        return JSONResponse({
            "source": "plantnet",
            "plant_name": plantnet_result["species"],
            "scientific_name": plantnet_result.get("scientific_name"),
            "score": plantnet_result.get("score"),
            "local_model_confidence": round(keras_result["confidence"], 4),
        })

    raise HTTPException(status_code=404, detail="Plant could not be identified.")