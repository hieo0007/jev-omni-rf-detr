# Jev-Omni e RF-DETR: pesquisa, testes e execução remota

Projeto de pesquisa para entender dois modelos de visão computacional, testá-los de verdade e descobrir se eles podem rodar num servidor acessível pela internet com o meu PC desligado, de graça ou barato.

Regra seguida em todo o projeto: **nada de resultado, link ou preço inventado**. Cada resultado abaixo diz se foi testado ou se é só pesquisa.

Última atualização: 14/10/2026.

---

## Sumário

1. [Os dois modelos em uma tabela](#1-os-dois-modelos-em-uma-tabela)
2. [Jev-Omni](#2-jev-omni)
3. [RF-DETR](#3-rf-detr)
4. [Resultados reais dos testes](#4-resultados-reais-dos-testes)
5. [Os dois juntos com a câmera do celular](#5-os-dois-juntos-com-a-câmera-do-celular)
6. [Hospedagem remota: opções e plano](#6-hospedagem-remota-opções-e-plano)
7. [Como reproduzir no Windows](#7-como-reproduzir-no-windows)
8. [Estrutura do repositório](#8-estrutura-do-repositório)
9. [Limitações e próximos passos](#9-limitações-e-próximos-passos)
10. [Fontes](#10-fontes)

---

## 1. Os dois modelos em uma tabela

| | Jev-Omni | RF-DETR |
|---|---|---|
| O que faz | Classificador de decisão multimodal: recebe um contexto, uma pergunta e opções (e, se quiser, uma imagem, áudio ou vídeo) e devolve a probabilidade de cada opção. **Não gera texto.** | Detector de objetos em tempo real: devolve caixas, classes (80 classes do COCO) e confiança. |
| Autor | akhilaaa3 (Hugging Face), projeto independente | Roboflow |
| Tamanho | ~12B parâmetros, checkpoint BF16 de 23,9 GB | 30,5M a 126,9M parâmetros; o Nano tem 349 MB |
| Precisa de GPU? | Sim, CUDA obrigatório (GPU grande, ~40–48 GB estimados) | Não. Roda bem em CPU |
| Licença | Apache-2.0 (segundo o model card) | Apache-2.0 (Nano a Large); XL e 2XL em PML 1.0 |
| Onde foi testado | No Space público do autor (servidor do Hugging Face), chamado do navegador e por script | No container de pesquisa (CPU) e no PC do Daniel |

Os dois são tratados como **ambientes separados**: têm requisitos e versões de dependências diferentes (o Jev-Omni fixa `transformers==5.17.0`; o `rfdetr` instalou `transformers 5.19.0`).

## 2. Jev-Omni

- Model card: https://huggingface.co/akhilaaa3/Jev-Omni
- Space de demonstração do autor: https://huggingface.co/spaces/akhilaaa3/jev-omni

**Como funciona.** É um fine-tune do `google/gemma-4-12B-it` com uma "cabeça" linear de 256 saídas. O modelo faz uma única passada, pega o estado oculto do último token e a cabeça transforma isso numa probabilidade por opção. Por isso é rápido e não escreve texto.

**Entradas:** `state` (contexto), `question` (pergunta), `options` (2 a 256 opções, qualidade comprovada até 20), `media` opcional e `modality` (`text`, `image`, `audio` ou `video`).

**Requisitos (do model card):** GPU CUDA, ~24 GB só de pesos, `torch>=2.10`, `transformers==5.17.0`, entre outros. Uma GPU de 16 GB (T4 do Colab grátis) ou de 24 GB (L4, A10G) provavelmente não basta; isso é estimativa, não teste.

**Atenção ao nome:** existem outros projetos parecidos (o "Jev" comercial da TypeSafe AI e o "OmniJev"). Este projeto usa o `akhilaaa3/Jev-Omni`, confirmado pelo Daniel.

## 3. RF-DETR

- Repositório: https://github.com/roboflow/rf-detr
- Documentação: https://rfdetr.roboflow.com/latest/
- PyPI: https://pypi.org/project/rfdetr/ (versão usada: 1.11.2)

**Como funciona.** Detector baseado em transformer (DETR) com backbone DINOv2. Recebe uma imagem e devolve as caixas (`xyxy`), a classe (`class_id`) e a confiança. Os modelos prontos reconhecem as 80 classes do COCO (pessoa, gato, xícara etc.); para outras classes é preciso treinar.

| Modelo | Parâmetros | COCO AP50:95 | Latência (T4, TensorRT FP16) | Licença |
|---|---|---|---|---|
| Nano | 30,5M | 48,4 | 2,3 ms | Apache 2.0 |
| Small | 32,1M | 53,0 | 3,5 ms | Apache 2.0 |
| Medium | 33,7M | 54,7 | 4,4 ms | Apache 2.0 |
| Large | 33,9M | 56,5 | 6,8 ms | Apache 2.0 |
| XL | 126,4M | 58,6 | 11,5 ms | PML 1.0 |
| 2XL | 126,9M | 60,1 | 17,2 ms | PML 1.0 |

(Números do README oficial, não medidos aqui.)

## 4. Resultados reais dos testes

Todos executados em 14/10/2026. As fotos de teste são as imagens de exemplo do scikit-image (`astronaut.jpg`, `chelsea.jpg`, `coffee.jpg`).

### RF-DETR em CPU (container de pesquisa: 4 vCPU Xeon 2,1 GHz, 15 GB RAM, Python 3.13, sem GPU)

| Modelo | Imagem | Detecções (limiar 0,5) | Tempo por imagem |
|---|---|---|---|
| Nano | astronaut | person 0,975 | 0,12–0,14 s |
| Nano | chelsea | cat 0,96 | 0,14–0,16 s |
| Nano | coffee | cup 0,941, spoon 0,903, dining table 0,87 | 0,11–0,12 s |
| Medium | coffee | cup 0,947, spoon 0,882, dining table 0,843 | 0,28–0,31 s |

- Carregar o modelo levou 3,4–8,1 s; a primeira inferência é mais lenta (0,21–0,40 s).
- **API HTTP** (`rf-detr/api.py`): recebeu `coffee.jpg` e respondeu HTTP 200 em 0,26 s com as mesmas 3 detecções. Testado em localhost.
- **App Gradio** (`hospedagem/hf-spaces/rf-detr/app.py`): chamado por `gradio_client` em localhost, devolveu cup 0,941, spoon 0,903, dining table 0,87. Ainda não testado dentro do Hugging Face.
- Pico de memória observado: ~1,2 GB de RAM.

### Jev-Omni remoto (Space do autor, ZeroGPU)

| Teste | Como | Pergunta | Resultado | Latência |
|---|---|---|---|---|
| Navegador | Interface web, logado | "Is there a person in the image?" (astronaut) | Yes 100% | 1049 ms |
| Script | `jev-omni/chamar_space_publico.py`, Windows, Python 3.13.16, gradio_client 2.7.2 | "Há uma pessoa na imagem?" | Sim 0,99966 / Não 0,00034 | 1056 ms |
| Script (negativo) | `jev-omni/teste_negativo.py` | "Há um gato na imagem?" | Não 0,99991 / Sim 0,0000875 | 1035 ms |

Conclusão: **o envio de imagem por script ao Jev-Omni remoto funciona**, mas usando o Space de um terceiro (o autor do modelo). O exemplo local (`jev-omni/exemplo_local.py`) **não foi testado**, porque exige uma GPU grande.

Observação: no Python 3.15 a instalação falhou (o `pyyaml` não tem pacote pronto para 3.15). Use **Python 3.13**.

## 5. Os dois juntos com a câmera do celular

`celular-juntos/app_celular.py` é um app Gradio que roda no PC:

1. O celular, na mesma rede Wi-Fi, abre `http://IP-DO-PC:7860` e tira uma foto.
2. O **RF-DETR** roda no PC (CPU) e desenha as caixas dos objetos.
3. A foto é enviada ao **Jev-Omni** no Space do autor, que classifica a imagem entre categorias (Pessoa, Animal, Comida, Veículo, Paisagem, Objeto).

**Testado em 14/10/2026:** foto tirada com iPhone, Jev-Omni respondeu "Objeto" 99,8% em ~1 s.

Limitações: o PC precisa estar ligado e o celular na mesma rede. O link público do Gradio (`share=True`) não funcionou no PC do Daniel, provavelmente por causa do Cloudflare WARP. Os dois modelos continuam separados; o app só junta os resultados.

## 6. Hospedagem remota: opções e plano

Preços e limites lidos nas páginas oficiais em 14/10/2026 (podem mudar). Nada foi contratado.

### Opções gratuitas

| Opção | Serve para | Limites | Situação |
|---|---|---|---|
| **Hugging Face ZeroGPU Space** | Os dois (um Space para cada) | Conta grátis com e-mail verificado e **mais de 30 dias** pode ter até 2 Spaces Gradio ZeroGPU. Cota de GPU de 5 min/dia por quem chama. O Space dorme sem uso. | **Plano escolhido.** A conta HF foi criada em 14/10/2026, então libera por volta de **13/11/2026**. |
| Space do autor (akhilaaa3/jev-omni) | Jev-Omni, para testes | Depende de terceiro; reserva 45 s de GPU por chamada | Em uso agora, funcionando |
| Oracle Cloud Always Free (ARM, 2 OCPU, 12 GB) | RF-DETR (CPU) | Exige cartão; VM ociosa pode ser recuperada | Tentada em 14/10/2026: cobrou ~R$ 5 de verificação e a conta não foi criada |
| Google Cloud Run (CPU) | RF-DETR | Cota mensal grátis; exige conta de faturamento | Não testado |

Render, Koyeb e Back4App (grátis, sem cartão) têm só 512 MB de RAM, menos que os ~1,2 GB que o RF-DETR usou. Colab e Kaggle não servem como servidor sempre acessível.

Na tela de criar Space da conta grátis aparece "Gradio e Docker exigem plano pago": isso vale para CPU sem GPU. A documentação oficial do ZeroGPU confirma a exceção grátis para contas com 30+ dias.

### Opções pagas (só estimativa, não contratar)

| Opção | Preço |
|---|---|
| Hugging Face PRO (ZeroGPU com 40 min/dia) | US$ 9/mês |
| Modal (L40S 48 GB / A100 40 GB) | US$ 1,95/h / US$ 2,10/h, cobrado por segundo |
| HF Spaces ou Inference Endpoints com L40S | US$ 1,80/h |

### Plano de implantação (a partir de ~13/11/2026)

1. Criar o Space **rf-detr** (Gradio + ZeroGPU) com os arquivos de `hospedagem/hf-spaces/rf-detr/`.
2. Duplicar o Space do autor `akhilaaa3/jev-omni` para a própria conta, mantendo ZeroGPU.
3. Com o PC desligado, chamar os dois Spaces do celular e registrar resultado e latência (acordado e "frio").

Passo a passo: [hospedagem/hf-spaces/GUIA.md](hospedagem/hf-spaces/GUIA.md). Alternativa para o RF-DETR sempre ligado: [hospedagem/oracle-rf-detr/GUIA.md](hospedagem/oracle-rf-detr/GUIA.md).

## 7. Como reproduzir no Windows

Abra o **Prompt de Comando (cmd)** na pasta do exemplo e rode um comando por vez.

### Jev-Omni pelo Space do autor

Crie um token de leitura em https://huggingface.co/settings/tokens. **Nunca cole o token em conversas, arquivos ou no GitHub**; se isso acontecer, revogue-o e crie outro.

```
cd jev-omni
py -3.13 -m pip install gradio_client
set HF_TOKEN=seu_token_aqui
py -3.13 chamar_space_publico.py
py -3.13 teste_negativo.py
```

Esperado: "Sim" com ~99,9% no primeiro e "Não" com ~99,9% no segundo.

### RF-DETR no seu PC

```
cd rf-detr
py -3.13 -m pip install rfdetr
py -3.13 teste_local.py coffee.jpg
```

Esperado: cup ~0,94, spoon ~0,90, dining table ~0,87. Na primeira vez ele baixa os pesos (349 MB).

Para testar a API HTTP:

```
py -3.13 -m pip install fastapi uvicorn python-multipart requests
py -3.13 -m uvicorn api:app --port 8000
```

Em outro cmd, na mesma pasta:

```
py -3.13 enviar_imagem.py http://localhost:8000 coffee.jpg
```

### Os dois juntos com o celular

```
cd celular-juntos
py -3.13 -m pip install rfdetr gradio gradio_client opencv-python-headless
set HF_TOKEN=seu_token_aqui
py -3.13 app_celular.py
```

Descubra o IP do PC com `ipconfig` (linha "Endereço IPv4") e abra `http://ESSE_IP:7860` no celular, na mesma rede Wi-Fi.

## 8. Estrutura do repositório

```
README.md                    este arquivo
docs/relatorio-pesquisa.md   relatório completo da pesquisa (14/10/2026)
jev-omni/                    exemplos do Jev-Omni (Space remoto e uso local com GPU)
rf-detr/                     exemplos do RF-DETR (teste local, API FastAPI e cliente)
celular-juntos/              app que junta os dois com a câmera do celular
hospedagem/hf-spaces/        guia e arquivos para os Spaces ZeroGPU do Hugging Face
hospedagem/oracle-rf-detr/   guia e script para o RF-DETR numa VM Oracle Always Free
```

## 9. Limitações e próximos passos

- Ainda **não** há execução comprovada com o PC desligado num servidor próprio. O Jev-Omni remoto funciona só via Space de terceiro; o RF-DETR ainda roda no PC.
- O container de pesquisa não tinha GPU e não alcançava o huggingface.co, então os testes do Hugging Face foram feitos no PC do Daniel.
- Os benchmarks do Jev-Omni são do autor e não foram verificados aqui.
- A regra exata de consumo da cota do ZeroGPU não está documentada oficialmente.
- Fora do escopo desta etapa: vídeo ao vivo e processamento em lote.

Próximo passo: por volta de 13/11/2026, criar os dois Spaces ZeroGPU e fazer o teste final com o PC desligado.

## 10. Fontes

- Jev-Omni: https://huggingface.co/akhilaaa3/Jev-Omni
- Space do Jev-Omni: https://huggingface.co/spaces/akhilaaa3/jev-omni
- RF-DETR: https://github.com/roboflow/rf-detr · https://rfdetr.roboflow.com/latest/ · https://pypi.org/project/rfdetr/
- ZeroGPU: https://huggingface.co/docs/hub/en/spaces-zerogpu
- Spaces: https://huggingface.co/docs/hub/en/spaces-overview
- Preços Hugging Face: https://huggingface.co/pricing
- Oracle Always Free: https://docs.oracle.com/en-us/iaas/Content/FreeTier/resourceref.htm
- Google Cloud Run: https://cloud.google.com/run/pricing
- Modal: https://modal.com/pricing
