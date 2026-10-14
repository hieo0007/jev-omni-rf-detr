# RF-DETR na Oracle Always Free (passo a passo)

Nada aqui foi testado na Oracle ainda. A API (`api.py`) foi testada em localhost.

1. **Conta**: https://www.oracle.com/cloud/free/ > "Start for free". Pede cartão para verificação. Escolha uma região próxima (ex.: Brazil East – São Paulo). A região não pode ser trocada depois.
2. **Criar a VM**: Compute > Instances > Create instance.
   - Image: **Canonical Ubuntu 24.04**.
   - Shape: **Ampere > VM.Standard.A1.Flex**, 2 OCPU, 12 GB (limite grátis atual).
   - Em "Add SSH keys", escolha "Generate a key pair" e **baixe a chave privada**.
   - Se aparecer "Out of capacity", tente outro "Availability domain" ou mais tarde.
3. **Abrir a porta 8000 na Oracle**: na página da instância > Subnet > Security List > Add Ingress Rule: Source `0.0.0.0/0`, TCP, porta de destino `8000`.
4. **Entrar na VM** (no PowerShell do seu PC):
   `ssh -i caminho\da\chave.key ubuntu@IP_PUBLICO`
5. **Enviar os arquivos** (em outro PowerShell, na pasta desta guia):
   `scp -i caminho\da\chave.key api.py instalar.sh ubuntu@IP_PUBLICO:~`
6. **Instalar** (dentro da VM): `bash instalar.sh` (demora alguns minutos).
7. **Testar com o PC desligado**: do celular ou de outro computador, ou do seu PC antes de desligar:
   `py -3.13 -m pip install requests` e depois `py -3.13 enviar_imagem.py http://IP_PUBLICO:8000 coffee.jpg`
   Esperado: cup ~0,94, spoon ~0,90, dining table ~0,87.
   Depois desligue o PC e acesse `http://IP_PUBLICO:8000/docs` pelo celular: dá para enviar uma imagem por essa página.

Cuidados: a Oracle pode recuperar VMs grátis ociosas por 7 dias (uso de CPU, rede e memória abaixo de 20%). A API fica aberta para qualquer pessoa que souber o IP.
