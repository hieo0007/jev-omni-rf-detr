# Envia uma imagem ao Space público akhilaaa3/jev-omni (ZeroGPU) e recebe as probabilidades.
# TESTADO em 14/10/2026 no PC do Daniel (Windows, Python 3.13.16, gradio_client 2.7.2):
#   saída: Sim 0.99966 / Não 0.00034, ~1056 ms (veja o README).
# Instalar:  py -3.13 -m pip install gradio_client
# Rodar:     set HF_TOKEN=seu_token   e depois   py -3.13 chamar_space_publico.py
# Endpoint "/decide" com parâmetros state, question, options_text, media, modality (confirmado por view_api()).
import os
from gradio_client import Client, handle_file

client = Client("akhilaaa3/jev-omni", token=os.getenv("HF_TOKEN"))  # com token, usa a sua cota diária de GPU
client.view_api()

probs, verdict = client.predict(
    "Foto enviada pelo usuário.",   # state
    "Há uma pessoa na imagem?",     # question
    "Sim\nNão",                      # options, uma por linha
    handle_file("astronaut.jpg"),   # media
    "image",                         # modality
    api_name="/decide",
)
print(probs)
print(verdict)
