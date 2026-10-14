import io
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES

app = FastAPI()
model = RFDETRNano()

@app.post("/detect")
async def detect(file: UploadFile = File(...), threshold: float = 0.5):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    det = model.predict(img, threshold=threshold)
    return {"detections": [
        {"label": COCO_CLASSES[int(c)], "confidence": round(float(s), 3), "box_xyxy": [round(float(v), 1) for v in b]}
        for c, s, b in zip(det.class_id, det.confidence, det.xyxy)]}
