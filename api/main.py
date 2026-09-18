from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from sentinel_vision.predict import Predictor
app=FastAPI(title="SentinelVision API",version="0.1.0")
_predictor=None

def get_predictor():
    global _predictor
    if _predictor is None: _predictor=Predictor()
    return _predictor

@app.get("/health")
def health():
    try: get_predictor(); return {"status":"ok","model_loaded":True}
    except FileNotFoundError: return {"status":"ok","model_loaded":False}

@app.post("/predict")
async def predict(file:UploadFile=File(...)):
    if not file.content_type or not file.content_type.startswith("image/"): raise HTTPException(415,"Upload an image file")
    try: image=Image.open(file.file)
    except Exception as exc: raise HTTPException(400,"Invalid image") from exc
    try: return get_predictor().predict(image)
    except FileNotFoundError as exc: raise HTTPException(503,str(exc))
