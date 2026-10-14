# Documentação técnica do projeto: Jev-Omni e RF-DETR

Relatório do estado real do projeto em **14/10/2026**. Cada item diz se foi **verificado** (executado, com resultado registrado) ou apenas **pesquisado/escrito** (existe o arquivo, mas não foi executado).

Repositório: https://github.com/hieo0007/jev-omni-rf-detr

---

## 1. Objetivo do projeto

Pesquisar, testar e documentar dois modelos de visão computacional, **Jev-Omni** (classificador de decisão multimodal) e **RF-DETR** (detector de objetos), e descobrir se ambos podem rodar num servidor acessível pela internet com o PC do Daniel desligado, de graça ou com baixo custo.

Primeira etapa (atual): exemplos funcionais e envio de uma imagem ao servidor. Fora do escopo por enquanto: vídeo ao vivo e processamento em lote.

Regras do projeto: não inventar resultados, links ou preços; tratar os dois modelos como ambientes separados; nunca publicar tokens.

## 2. Origem e histórico

- O projeto nasceu como pesquisa dentro de um projeto do Claude, com os arquivos na pasta compartilhada `pesquisa/`. **Não havia Git nem repositório remoto** antes de 14/10/2026.
- Em 14/10/2026 o conteúdo foi organizado nesta estrutura e publicado neste repositório (primeiro commit).
- Não existe frontend próprio, backend de produção nem banco de dados. Os "servidores" são exemplos pequenos (FastAPI e Gradio).

## 3. Estrutura de pastas

```
README.md                         visão geral, resultados e como reproduzir
DOCUMENTACAO_PROJETO.md           este relatório técnico
.env.example                      modelo da variável HF_TOKEN (sem valor)
.gitignore                        ignora .env, venv, __pycache__, pesos de modelo
docs/relatorio-pesquisa.md        relatório completo da pesquisa (fontes, preços, hardware)
jev-omni/
  chamar_space_publico.py         envia imagem ao Space do autor (pergunta positiva)
  teste_negativo.py               mesma imagem, pergunta negativa
  exemplo_local.py                uso local oficial (exige GPU ~24 GB+)
  requirements.txt                gradio_client
  *.jpg                           imagens de teste
rf-detr/
  teste_local.py                  inferência em CPU e medição de tempo
  api.py                          API FastAPI: POST /detect
  enviar_imagem.py                cliente HTTP da API
  requirements.txt                rfdetr==1.11.2, fastapi, uvicorn, python-multipart, requests
  *.jpg
celular-juntos/
  app_celular.py                  app Gradio: câmera do celular -> RF-DETR (local) + Jev-Omni (Space)
  requirements.txt
hospedagem/hf-spaces/
  GUIA.md                         passo a passo dos 2 Spaces ZeroGPU
  rf-detr/                        app.py, requirements.txt, README.md (cabeçalho do Space), cliente
hospedagem/oracle-rf-detr/
  GUIA.md                         passo a passo da VM Oracle Always Free
  instalar.sh                     instala a API e cria serviço systemd
  api.py, enviar_imagem.py        cópias dos arquivos de rf-detr/
```

## 4. Arquitetura

| Camada | Situação |
|---|---|
| Frontend | Não há frontend próprio. As interfaces são geradas pelo Gradio (`celular-juntos/app_celular.py`, `hospedagem/hf-spaces/rf-detr/app.py`) e pela página `/docs` do FastAPI. |
| Backend / APIs | `rf-detr/api.py` (FastAPI, `POST /detect`, multipart `file`, parâmetro `threshold`). App Gradio do RF-DETR com endpoint `/detect`. Jev-Omni consumido como API remota do Space `akhilaaa3/jev-omni`, endpoint `/decide(state, question, options_text, media, modality)`. |
| Banco de dados | Nenhum. Nada é armazenado. |
| Serviços externos | Hugging Face Spaces (ZeroGPU) para o Jev-Omni; download automático dos pesos do RF-DETR (349 MB) na primeira execução. |
| Autenticação | Token de leitura do Hugging Face, lido da variável de ambiente `HF_TOKEN`. A API FastAPI **não tem autenticação**. |

## 5. Tecnologias

- Linguagem: **Python 3.13** (o 3.15 falhou no Windows ao instalar `pyyaml`).
- RF-DETR: `rfdetr 1.11.2`, `torch 2.14.1` (CPU), `transformers 5.19.0`, `supervision 0.30.9`.
- Servidores: `fastapi 0.143.1` + `uvicorn`, `gradio 6.30.0`.
- Cliente remoto: `gradio_client 2.7.2`.
- Jev-Omni (só no exemplo local, não executado): `transformers==5.17.0`, `torch>=2.10`, GPU CUDA.
- Shell: `instalar.sh` (bash, Ubuntu 24.04 ARM, systemd, iptables).

Versões acima são as instaladas e usadas nos testes de 14/10/2026.

## 6. O que foi verificado (executado)

### 6.1 Testes refeitos nesta análise (14/10/2026, ~23:20 UTC)

Ambiente: container Linux, 4 vCPU, 15 GB RAM, Python 3.13.16, sem GPU, sem acesso ao huggingface.co.

| Teste | Comando | Resultado |
|---|---|---|
| Sintaxe de todos os `.py` | `python -m py_compile` | OK |
| Sintaxe do `instalar.sh` | `bash -n` | OK (não executado) |
| RF-DETR Nano, coffee.jpg | `python teste_local.py coffee.jpg` | cup 0,941, spoon 0,903, dining table 0,87; 0,12–0,14 s/imagem |
| RF-DETR Nano, astronaut.jpg | idem | person 0,975; 0,15–0,17 s |
| RF-DETR Nano, chelsea.jpg | idem | cat 0,96; 0,12–0,14 s |
| API FastAPI | `uvicorn api:app` + `python enviar_imagem.py http://localhost:8000 coffee.jpg` | HTTP 200, mesmas 3 detecções, 0,39 s no total |
| App Gradio do Space RF-DETR | `python app.py` + `python enviar_imagem.py http://127.0.0.1:7860 coffee.jpg` | mesmas 3 detecções e imagem anotada |

Os resultados batem exatamente com os de mais cedo no mesmo dia.

### 6.2 Testes feitos pelo Daniel no PC dele (Windows, 14/10/2026)

| Teste | Resultado |
|---|---|
| Jev-Omni via navegador no Space do autor ("Is there a person?", astronaut) | Yes 100%, 1049 ms |
| `jev-omni/chamar_space_publico.py` | Sim 0,99966 / Não 0,00034, 1056 ms |
| `jev-omni/teste_negativo.py` ("Há um gato?") | Não 0,99991 / Sim 0,0000875, 1035 ms |
| `celular-juntos/app_celular.py` com iPhone na mesma rede Wi-Fi | Jev-Omni "Objeto" 99,8%, ~1 s; RF-DETR desenhou as caixas |

## 7. Estado das funcionalidades

### Concluídas e funcionando
- Detecção de objetos com RF-DETR em CPU (script, API HTTP e app Gradio), em localhost.
- Envio de imagem por script ao Jev-Omni remoto e leitura das probabilidades (via Space de terceiro).
- App que junta os dois com a câmera do celular, na rede local.

### Parcialmente implementadas
- **Space próprio do RF-DETR** (`hospedagem/hf-spaces/rf-detr/`): código pronto e testado localmente; **não** foi publicado no Hugging Face (a conta precisa ter 30 dias para ZeroGPU grátis, previsto para ~13/11/2026).
- **Space próprio do Jev-Omni**: plano é duplicar o Space do autor; ainda não feito pelo mesmo motivo.
- **RF-DETR na Oracle** (`hospedagem/oracle-rf-detr/`): guia e script prontos; **não** executados (a conta Oracle não foi criada).

### Não desenvolvidas
- Execução com o PC desligado num servidor próprio (objetivo final): ainda não comprovada.
- Jev-Omni rodando localmente (`exemplo_local.py`): não executado; exige GPU de ~40–48 GB (estimativa).
- Vídeo ao vivo, processamento em lote, autenticação na API, testes automatizados.

## 8. Problemas técnicos e pendências conhecidas

1. **Dependência de um Space de terceiro** para o Jev-Omni: se o autor desligar ou mudar o Space, os scripts param de funcionar.
2. **Cota de GPU**: o Space do autor reserva 45 s de ZeroGPU por chamada; a conta grátis tem 5 min/dia, o que pode dar poucas chamadas por dia (a regra exata de consumo não está documentada oficialmente).
3. **`app_celular.py`**: o cliente do Jev-Omni é criado ao iniciar o app; se o Hugging Face estiver inacessível, o app inteiro não abre (o tratamento de erro só cobre a chamada). Usa `share=True`, que falhou no PC do Daniel (provavelmente pelo Cloudflare WARP); o acesso funciona pelo IP local.
4. **API sem autenticação**: em servidor público, qualquer pessoa com o IP pode usar.
5. **Arquivos duplicados**: `hospedagem/oracle-rf-detr/api.py` e `enviar_imagem.py` são cópias dos de `rf-detr/` (mantidos para o guia da Oracle ser autossuficiente).
6. **Sem testes automatizados**: a verificação é manual, comparando com os resultados esperados (cup ~0,94, spoon ~0,90, dining table ~0,87).
7. **Token do Hugging Face já foi colado em conversas**: deve ser revogado e recriado em https://huggingface.co/settings/tokens. Nenhum token está neste repositório (verificado).

## 9. Como executar localmente (Windows, cmd)

Pré-requisitos: Python 3.13 (`py -3.13 --version`), Git opcional.

```
git clone https://github.com/hieo0007/jev-omni-rf-detr.git
cd jev-omni-rf-detr
```

RF-DETR:
```
cd rf-detr
py -3.13 -m pip install -r requirements.txt
py -3.13 teste_local.py coffee.jpg
```

Jev-Omni (remoto):
```
cd jev-omni
py -3.13 -m pip install -r requirements.txt
set HF_TOKEN=seu_token_de_leitura
py -3.13 chamar_space_publico.py
```

App do celular: veja a seção 7 do README.

## 10. Variáveis de ambiente sem expor segredos

- A única variável é `HF_TOKEN` (token de leitura do Hugging Face). Os scripts leem com `os.getenv("HF_TOKEN")`; nada fica no código.
- No Windows, defina só na janela do cmd: `set HF_TOKEN=...`. Ela some ao fechar a janela.
- `.env.example` mostra o nome da variável sem valor. Um `.env` real está no `.gitignore` e nunca deve ser enviado.
- No Hugging Face Space, tokens vão em **Settings > Variables and secrets > New secret**, nunca no código.
- Na Oracle, se um dia precisar, use `Environment=` num arquivo de serviço fora do repositório ou o OCI Vault; nunca faça commit.

## 11. Preparação para Oracle Cloud (OCI)

### O que já existe
- `hospedagem/oracle-rf-detr/GUIA.md`: criar VM Always Free (Ampere A1, Ubuntu 24.04, 2 OCPU, 12 GB), abrir a porta 8000 na Security List, enviar arquivos por `scp`.
- `hospedagem/oracle-rf-detr/instalar.sh`: cria venv, instala `torch torchvision rfdetr fastapi uvicorn python-multipart`, libera a porta 8000 no iptables e cria o serviço `rfdetr` no systemd (sobe sozinho e reinicia se cair).
- Nenhum recurso foi criado na Oracle. Nenhuma credencial OCI existe no projeto.

### O que cabe na OCI
- **RF-DETR: sim.** Roda em CPU, usou ~1,2 GB de RAM; a VM grátis tem 12 GB.
- **Jev-Omni: não na camada grátis.** Exige GPU CUDA grande; as GPUs da OCI são pagas. Continua no Hugging Face ZeroGPU.
- Não há banco de dados a migrar.

### O que falta para uma implantação futura
1. Criar a conta OCI (a tentativa de 14/10/2026 não concluiu; a Oracle cobrou ~R$ 5 de verificação). Escolher a região uma vez (não muda depois).
2. Seguir o GUIA: VM, chave SSH, regra de entrada da porta 8000, `instalar.sh`.
3. Validar: `py -3.13 enviar_imagem.py http://IP_PUBLICO:8000 coffee.jpg` deve dar cup ~0,94, spoon ~0,90, dining table ~0,87.
4. Adaptações recomendadas antes de deixar público:
   - Autenticação simples na API (ex.: cabeçalho com chave lida de variável de ambiente) ou restringir a Security List ao seu IP.
   - HTTPS (ex.: Caddy como proxy reverso com domínio) se for usar fora de testes.
   - Fixar versões (`rfdetr==1.11.2`; `torch` com wheel ARM do PyPI) para evitar quebra em reinstalações.
5. Cuidados de custo: usar só o shape **VM.Standard.A1.Flex** dentro do limite Always Free; não criar volumes, IPs reservados ou load balancers extras; a Oracle pode recuperar VMs grátis ociosas por 7 dias.

Não verificado: `instalar.sh` nunca rodou numa VM ARM real.

## 12. PENDÊNCIAS E PRÓXIMAS ETAPAS

Em ordem de prioridade:

1. **Revogar o token do Hugging Face** que já foi colado em conversas e criar um novo (segurança).
2. **~13/11/2026: criar o Space ZeroGPU do RF-DETR** com `hospedagem/hf-spaces/rf-detr/` e testar com `enviar_imagem.py SEU_USUARIO/rf-detr coffee.jpg`.
3. **~13/11/2026: duplicar o Space do Jev-Omni** na sua conta e trocar `akhilaaa3/jev-omni` pelo seu Space nos scripts.
4. **Teste final com o PC desligado**: chamar os dois Spaces pelo celular; registrar resultado e latência (Space acordado e "frio").
5. **Ajustar `app_celular.py`** para abrir mesmo sem acesso ao Hugging Face e, depois do passo 3, apontar para os Spaces próprios (então ele pode virar um Space e rodar sem o PC).
6. **Opcional, RF-DETR sempre ligado na Oracle**: criar conta e seguir a seção 11, com autenticação na API.
7. Adicionar um teste automatizado simples (pytest) que roda o RF-DETR no `coffee.jpg` e confere as classes.
8. Avaliar plano pago só se a cota grátis não bastar: HF PRO US$ 9/mês (40 min/dia de ZeroGPU). Não contratar sem decisão.

## 13. Como compartilhar este repositório

- Repositório: https://github.com/hieo0007/jev-omni-rf-detr
- Visibilidade no momento da publicação: **público** (qualquer pessoa pode ver e clonar sem ser colaborador).
- Para torná-lo privado: Settings > General > Danger Zone > Change visibility > Make private.
- Para dar acesso a alguém (necessário se ficar privado, ou para a pessoa poder enviar alterações): Settings > Collaborators > Add people > digite o usuário do GitHub da pessoa. Ela recebe um convite por e-mail e precisa aceitar.

## 14. O que outra pessoa precisa para clonar e executar

- Python 3.13 e, opcionalmente, Git.
- Internet para baixar os pesos do RF-DETR (349 MB) na primeira execução.
- Uma conta Hugging Face e o próprio token de leitura para os scripts do Jev-Omni.
- Nenhuma GPU para os exemplos que funcionam hoje.
