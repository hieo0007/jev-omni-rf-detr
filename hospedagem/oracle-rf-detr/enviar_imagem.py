# Cliente que envia uma imagem para a API do api.py e imprime as detecções.
# Testado só em localhost dentro do container de pesquisa (ver relatório).
# Uso: python enviar_imagem.py http://localhost:8000 coffee.jpg
import sys
import requests

url, image = sys.argv[1], sys.argv[2]
with open(image, "rb") as f:
    r = requests.post(f"{url}/detect", files={"file": f}, params={"threshold": 0.5}, timeout=60)
r.raise_for_status()
print(r.json())
