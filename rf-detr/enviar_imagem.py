# Cliente que envia uma imagem para a API do api.py e imprime as detecções.
# Testado em localhost (HTTP 200, mesmas 3 detecções do teste local). Instalar: pip install requests
# Uso: python enviar_imagem.py http://localhost:8000 coffee.jpg
import sys
import requests

url, image = sys.argv[1], sys.argv[2]
with open(image, "rb") as f:
    r = requests.post(f"{url}/detect", files={"file": f}, params={"threshold": 0.5}, timeout=60)
r.raise_for_status()
print(r.json())
