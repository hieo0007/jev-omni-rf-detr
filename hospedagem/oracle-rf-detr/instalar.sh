#!/bin/bash
# Instala a API do RF-DETR numa VM Ubuntu (Oracle Ampere A1, ARM). Rodar como: bash instalar.sh
# NÃO TESTADO na Oracle; a API em si foi testada em localhost (ver relatório).
set -e
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip
mkdir -p ~/rfdetr && cp api.py ~/rfdetr/ && cd ~/rfdetr
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install torch torchvision rfdetr fastapi uvicorn python-multipart
# Libera a porta 8000 no firewall do Ubuntu da Oracle (que bloqueia tudo por padrão)
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
sudo netfilter-persistent save || true
# Serviço que sobe sozinho e reinicia se cair
sudo tee /etc/systemd/system/rfdetr.service >/dev/null <<UNIT
[Unit]
Description=RF-DETR API
After=network.target
[Service]
User=$USER
WorkingDirectory=$HOME/rfdetr
ExecStart=$HOME/rfdetr/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload && sudo systemctl enable --now rfdetr
echo "Pronto. Teste: curl -F file=@coffee.jpg http://IP_PUBLICO:8000/detect"
