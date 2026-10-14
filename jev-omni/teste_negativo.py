# Teste negativo: mesma foto do astronauta, perguntando se há um gato (esperado: Não).
# TESTADO em 14/10/2026 no PC do Daniel: Não 0.99991 / Sim 0.0000875, ~1035 ms (veja o README).
# Instalar:  py -3.13 -m pip install gradio_client
# Rodar:     set HF_TOKEN=seu_token   e depois   py -3.13 teste_negativo.py
import os
from gradio_client import Client, handle_file

client = Client("akhilaaa3/jev-omni", token=os.getenv("HF_TOKEN"))  # com token, usa a sua cota diária de GPU
# client.view_api()

probs, verdict = client.predict(
    "Foto enviada pelo usuário.",   # state
    "Há um gato na imagem?",     # question
    "Sim\nNão",                      # options, uma por linha
    handle_file("astronaut.jpg"),   # media
    "image",                         # modality
    api_name="/decide",
)
print(probs)
print(verdict)
