# Space do RF-DETR: recebe uma imagem, devolve a imagem anotada e as detecções em JSON.
# Pensado para Hugging Face ZeroGPU (Gradio). Também roda localmente sem GPU: python app.py
import numpy as np
import gradio as gr
import supervision as sv
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES

try:
    import spaces  # disponível só dentro do Hugging Face Spaces
    gpu = spaces.GPU(duration=10)
except ImportError:
    gpu = lambda f: f

model = RFDETRNano()  # inferência em CPU (~0,1–0,3 s/imagem nos testes); o decorador só satisfaz o ZeroGPU


@gpu
def detect(image: Image.Image, threshold: float):
    image = image.convert("RGB")
    det = model.predict(image, threshold=threshold)
    labels = [f"{COCO_CLASSES[int(c)]} {s:.2f}" for c, s in zip(det.class_id, det.confidence)]
    frame = np.array(image)
    frame = sv.BoxAnnotator().annotate(frame, det)
    frame = sv.LabelAnnotator().annotate(frame, det, labels)
    result = [
        {"label": COCO_CLASSES[int(c)], "confidence": round(float(s), 3), "box_xyxy": [round(float(v), 1) for v in b]}
        for c, s, b in zip(det.class_id, det.confidence, det.xyxy)
    ]
    return Image.fromarray(frame), result


demo = gr.Interface(
    fn=detect,
    inputs=[gr.Image(type="pil", label="Imagem"), gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Limiar")],
    outputs=[gr.Image(label="Detecções"), gr.JSON(label="Resultado")],
    title="RF-DETR Nano",
    api_name="detect",
)

if __name__ == "__main__":
    demo.launch()
