# Envia uma imagem ao Space do RF-DETR e imprime as detecções.
# Uso: py -3.13 enviar_imagem.py SEU_USUARIO/rf-detr coffee.jpg
#  (ou http://127.0.0.1:7860 para testar localmente)
import sys
from gradio_client import Client, handle_file

client = Client(sys.argv[1])
annotated, detections = client.predict(handle_file(sys.argv[2]), 0.5, api_name="/detect")
print(detections)
print("imagem anotada salva em:", annotated)
