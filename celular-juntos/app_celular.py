# Foto da câmera do celular -> RF-DETR (onde estão os objetos) + Jev-Omni (classificação da imagem).
# Os dois modelos continuam separados: o RF-DETR roda neste PC (CPU) e o Jev-Omni roda no Space
# público akhilaaa3/jev-omni (Hugging Face), chamado pela internet. Este app só junta os resultados.
# Foto por foto; não é vídeo ao vivo.
#
# Instalar (Python 3.13):  py -3.13 -m pip install rfdetr gradio gradio_client opencv-python-headless
# Rodar:                    set HF_TOKEN=seu_token   e depois   py -3.13 app_celular.py
# No celular (mesma rede Wi-Fi do PC), abra http://IP-DO-PC:7860 e toque na câmera.
# TESTADO em 14/10/2026: foto de iPhone -> Jev-Omni "Objeto" 99,8% em ~1 s. O link público (share=True)
# não funcionou no PC do Daniel (provavelmente por causa do Cloudflare WARP); use o IP local.
import os
import numpy as np
import gradio as gr
import supervision as sv
from PIL import Image
from gradio_client import Client, handle_file
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES

CATEGORIAS = "Pessoa\nAnimal\nComida\nVeículo\nPaisagem\nObjeto"

detector = RFDETRNano()
jev = Client("akhilaaa3/jev-omni", token=os.getenv("HF_TOKEN"))


def analisar(foto: Image.Image, categorias: str):
    foto = foto.convert("RGB")

    # 1) RF-DETR: encontra os objetos e onde estão
    det = detector.predict(foto, threshold=0.5)
    nomes = [COCO_CLASSES[int(c)] for c in det.class_id]
    rotulos = [f"{n} {s:.2f}" for n, s in zip(nomes, det.confidence)]
    quadro = sv.BoxAnnotator().annotate(np.array(foto), det)
    quadro = sv.LabelAnnotator().annotate(quadro, det, rotulos)

    # 2) Jev-Omni: classifica a imagem entre as categorias (usa os objetos achados como contexto)
    caminho = "foto_celular.jpg"
    foto.save(caminho)
    contexto = "Foto tirada com a câmera do celular. Objetos detectados: " + (", ".join(nomes) or "nenhum")
    try:
        probs, resumo = jev.predict(contexto, "Qual categoria descreve melhor esta imagem?",
                                    categorias, handle_file(caminho), "image", api_name="/decide")
        classificacao = {c["label"]: c["confidence"] for c in probs["confidences"]}
    except Exception as erro:  # ex.: cota diária de ZeroGPU esgotada; o RF-DETR continua aparecendo
        classificacao = {}
        resumo = f"**Jev-Omni indisponível agora:** {erro}\n\nRF-DETR achou: {', '.join(rotulos) or 'nada'}"
    return Image.fromarray(quadro), classificacao, resumo


demo = gr.Interface(
    fn=analisar,
    inputs=[gr.Image(type="pil", sources=["upload", "webcam"], label="Foto (no celular, toque para usar a câmera)"),
            gr.Textbox(CATEGORIAS, lines=6, label="Categorias, uma por linha")],
    outputs=[gr.Image(label="RF-DETR: objetos encontrados"),
             gr.Label(label="Jev-Omni: classificação"),
             gr.Markdown()],
    title="Câmera do celular: RF-DETR + Jev-Omni",
)

if __name__ == "__main__":
    # server_name="0.0.0.0" deixa o celular na mesma rede Wi-Fi abrir http://IP-DO-PC:7860
    demo.launch(share=True, server_name="0.0.0.0")  # share cria link público; se falhar, use o IP local
