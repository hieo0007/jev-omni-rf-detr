# Exemplo mínimo do Jev-Omni (akhilaaa3/Jev-Omni), baseado no model card e no example.py oficiais.
# NÃO TESTADO: exige GPU CUDA com memória para o checkpoint BF16 de ~24 GB.
# Instalação: pip install -r requirements.txt (do repositório no Hugging Face); ffmpeg só para áudio.
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("akhilaaa3/Jev-Omni")  # baixa ~24 GB
sys.path.insert(0, path)
from jev_omni import load_jev_omni

classifier = load_jev_omni()

# Texto
print(classifier.predict(
    state="The meeting starts at 10 AM. It is now 9 AM.",
    question="Has the meeting started?",
    options=["Yes", "No"],
))

# Imagem: o card manda acrescentar media=<caminho> e modality="image"
print(classifier.predict(
    state="Foto enviada pelo usuário.",
    question="Há uma pessoa na imagem?",
    options=["Sim", "Não"],
    media="astronaut.jpg",
    modality="image",
))
