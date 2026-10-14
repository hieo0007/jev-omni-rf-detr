> Relatório de pesquisa original de 14/10/2026, mantido como registro. Algumas partes do início (por exemplo "Testado aqui? Não" e o pedido de confirmação do Jev-Omni) foram escritas antes dos testes feitos no mesmo dia; os resultados finais estão nas seções 7.1 e 7.2 e no [README](../README.md).

# Jev-Omni e RF-DETR: pesquisa, testes e viabilidade de execução remota

Data: 14/10/2026. Etapa: pesquisa (nada foi implantado em servidor nem contratado).

## Resumo

| | Jev-Omni | RF-DETR |
|---|---|---|
| O que faz | Classificador de decisão multimodal: recebe contexto + pergunta + opções (e opcionalmente uma imagem, áudio ou vídeo) e devolve uma probabilidade por opção. Não gera texto. | Detector de objetos em tempo real (também segmentação e keypoints): devolve caixas, classes e confiança. |
| Tamanho | ~12B parâmetros, checkpoint BF16 de 23,9 GB | 30,5M a 126,9M parâmetros; peso do Nano = 349 MB |
| Precisa de GPU? | Sim, CUDA obrigatório | Não. Rodou em CPU neste teste |
| Licença | Apache-2.0 (segundo o card, seguindo o Gemma 4; direitos do dataset à parte) | Apache-2.0 para N/S/M/L, segmentação e keypoints; XL e 2XL em PML 1.0 |
| Testado aqui? | **Não** (sem GPU e huggingface.co bloqueado neste container) | **Sim**, em CPU, incluindo envio de imagem por HTTP em localhost |
| Hospedagem grátis viável? | Provavelmente, via ZeroGPU do Hugging Face, com cota de poucos minutos de GPU por dia. Não comprovado. | Sim em teoria (ZeroGPU, Oracle Always Free, Cloud Run). Não comprovado em servidor remoto. |

**Ainda não há comprovação de execução remota para nenhum dos dois.** O que foi comprovado é só o RF-DETR rodando em CPU e respondendo por HTTP dentro do container de pesquisa.

---

## 1. Qual é o "Jev-Omni"? (precisa da sua confirmação)

Encontrei três tecnologias com nomes parecidos:

1. **Jev-Omni, de akhilaaa3 (Hugging Face)**: o único com esse nome exato. É o candidato que assumi.
   - Model card: https://huggingface.co/akhilaaa3/Jev-Omni
   - Space de demonstração oficial: https://huggingface.co/spaces/akhilaaa3/jev-omni
   - Dataset de avaliação: https://huggingface.co/datasets/akhilaaa3/decision-bench
   - Não existe repositório no GitHub; o código (`jev_omni.py`, `example.py`, `requirements.txt`) fica no próprio repositório do Hugging Face.
   - O card diz que é um projeto independente, **sem relação** com a TypeSafe AI nem com o modelo Jev dela.
2. **Jev, da TypeSafe AI**: modelo de "decisão" comercial, só texto, acessado por API paga. Não é "Omni" e não tem pesos públicos mencionados. Docs: https://docs.typesafe.ai/introduction · artigo: https://nexos.ai/blog/what-is-jev/
3. **OmniJev** (tinnel123): outro projeto, modelos visuais de 0,8B/2B/4B sobre Qwen3.5, Apache-2.0. Código: https://github.com/tinnel123666888/OmniJev (segundo a página https://jev-ai.pro/model/omnijev; não abri o GitHub).

Observação: sites como jev-ai.pro e thejevai.com são páginas comerciais de terceiros que repetem o card; usei o card do Hugging Face como fonte principal.

## 2. Jev-Omni (akhilaaa3/Jev-Omni)

### Como funciona
Fine-tune do `google/gemma-4-12B-it` (30 mil perguntas) com uma "cabeça" linear de 256 saídas (`head.pt`). Lendo o `app.py` do Space: o modelo faz um único forward pass, captura o estado oculto do último token e a cabeça converte em uma probabilidade por opção (softmax). Por isso não gera texto e é rápido.

- Entradas: `state` (contexto), `question`, `options` (2 a 256; qualidade só estabelecida até 20), `media` opcional e `modality` = text/image/audio/video.
- Vídeo é amostrado em 16 quadros; áudio limitado a 30 s (precisa de ffmpeg).
- Saída: opção escolhida, índice, confiança e probabilidade de cada opção.

### Requisitos
- GPU CUDA obrigatória. Arquivo `model.safetensors` de 23,9 GB (total do repositório: 24 GB) + memória extra em tempo de execução.
- `requirements.txt` oficial: `torch>=2.10`, `torchvision`, `transformers==5.17.0`, `accelerate`, `huggingface_hub`, `safetensors`, `numpy`, `Pillow`, `opencv-python-headless`, `soundfile`, `librosa`.
- Conclusão inferida (não testada): GPUs de 16 GB (T4 do Colab grátis) e de 24 GB (L4, A10G) não devem bastar para os pesos em BF16 com folga; o mínimo realista é uma GPU de ~40–48 GB (A100 40 GB, L40S, metade de RTX Pro 6000).
- Velocidade publicada no card (H200, aquecida): ~26 ms por imagem, ~83 ms para ~2k tokens de texto, ~504 ms para vídeo de 16 quadros.

### Resultados publicados pelo autor (não verificados por mim)
DecisionBench Medium 87,57%; JevBench 86,15%; MMAU 63,10% (micro); MVBench 53,10%; ECE 0,0400.

### Exemplo de código
- `exemplos/jev-omni/exemplo_local.py`: uso oficial (`load_jev_omni()` + `predict`), com texto e imagem. **Não testado.**
- `exemplos/jev-omni/chamar_space_publico.py`: envia uma imagem ao Space público via `gradio_client`. **Não testado** (este container não alcança huggingface.co).

## 3. RF-DETR (Roboflow)

### Fontes
- Repositório oficial: https://github.com/roboflow/rf-detr
- Documentação: https://rfdetr.roboflow.com/latest/
- PyPI: https://pypi.org/project/rfdetr/ (versão atual 1.11.2, Python >= 3.10, licença Apache 2.0)

### Como funciona
Detector baseado em transformer (DETR) com backbone DINOv2; tamanhos encontrados por busca de arquitetura. Recebe uma imagem e devolve um `supervision.Detections` com `xyxy`, `class_id` e `confidence`. Os modelos pré-treinados reconhecem as 80 classes do COCO; para classes próprias é preciso fine-tuning.

### Tamanhos de detecção (README oficial; latência em T4 com TensorRT FP16)
| Modelo | Parâmetros | COCO AP50:95 | Latência | Resolução | Licença |
|---|---|---|---|---|---|
| Nano | 30,5M | 48,4 | 2,3 ms | 384 | Apache 2.0 |
| Small | 32,1M | 53,0 | 3,5 ms | 512 | Apache 2.0 |
| Medium | 33,7M | 54,7 | 4,4 ms | 576 | Apache 2.0 |
| Large | 33,9M | 56,5 | 6,8 ms | 704 | Apache 2.0 |
| XL | 126,4M | 58,6 | 11,5 ms | 700 | PML 1.0 (`rfdetr[plus]`) |
| 2XL | 126,9M | 60,1 | 17,2 ms | 880 | PML 1.0 (`rfdetr[plus]`) |

Dependências principais: `torch>=2.2`, `torchvision>=0.17`, `transformers>=5.1,<6`, `supervision>=0.29`, `pydantic`. Há extras para ONNX, TensorRT, OpenVINO, CoreML e outros.

### Teste real executado (container de pesquisa, sem GPU)
Ambiente: 4 vCPU Intel Xeon 2,10 GHz, 15 GB RAM, Python 3.13.16, torch 2.14.1 (CPU), rfdetr 1.11.2. Pesos baixados automaticamente (349 MB, MD5 validado). Imagens de teste: fotos de exemplo do pacote scikit-image (`astronaut`, `chelsea`, `coffee`).

| Modelo | Imagem | Detecções (limiar 0,5) | Tempo por imagem (após aquecer) |
|---|---|---|---|
| Nano | astronaut | person 0,975 | 0,12–0,14 s |
| Nano | chelsea | cat 0,96 | 0,14–0,16 s |
| Nano | coffee | cup 0,941, spoon 0,903, dining table 0,87 | 0,11–0,12 s |
| Medium | coffee | cup 0,947, spoon 0,882, dining table 0,843 | 0,28–0,31 s |

Primeira inferência mais lenta (0,21–0,40 s); carregar o modelo levou 3,4–8,1 s.

**Envio de imagem por HTTP (somente localhost):** `exemplos/rf-detr/api.py` (FastAPI) recebeu `coffee.jpg` via `curl -F file=@coffee.jpg` e respondeu HTTP 200 em 0,26 s com as mesmas 3 detecções. Isto prova o exemplo, **não** a execução em servidor remoto.

### Arquivos
- `exemplos/rf-detr/teste_local.py`: teste mínimo (`python teste_local.py imagem.jpg`).
- `exemplos/rf-detr/api.py`: API `POST /detect` (`uvicorn api:app`).
- `exemplos/rf-detr/enviar_imagem.py`: cliente que envia a imagem e imprime o JSON.

## 4. Diferenças que importam para a hospedagem

- Jev-Omni é ~400 vezes maior em parâmetros que o RF-DETR Nano e exige GPU grande; RF-DETR roda em CPU comum.
- Versões de dependências diferentes: Jev-Omni fixa `transformers==5.17.0`; o rfdetr instalou `transformers 5.19.0`. Mantê-los em ambientes separados (como você pediu) evita conflito.
- Tarefas diferentes: RF-DETR diz "onde está o quê"; Jev-Omni responde perguntas de múltipla escolha sobre o conteúdo.

## 5. Opções de hospedagem (preços lidos nas páginas oficiais em 14/10/2026)

### Gratuitas
| Opção | Serve para | Limites relevantes | Fonte |
|---|---|---|---|
| **Hugging Face ZeroGPU Space** | Ambos (um Space para cada) | Contas grátis podem hospedar até 2 Spaces Gradio em ZeroGPU (e-mail verificado, conta com mais de 30 dias). GPU = metade de RTX Pro 6000 (48 GB) por padrão. Cota de GPU de quem chama: 5 min/dia (conta grátis), 2 min/dia (sem login). Chamada limitada a 60 s por padrão. O Space dorme sem uso. Só Gradio. | https://huggingface.co/docs/hub/en/spaces-zerogpu · https://huggingface.co/docs/hub/en/spaces-overview |
| Space público do autor (akhilaaa3/jev-omni) | Jev-Omni, só para testar | Já roda em ZeroGPU; reserva 45 s de GPU por chamada (90 s para vídeo). Se a cota for cobrada pela duração reservada (relato da comunidade, não confirmado oficialmente), seriam ~6 imagens/dia com conta grátis. Depende de terceiro. | app.py do Space |
| Oracle Cloud Always Free (Ampere A1) | RF-DETR (CPU) | 2 OCPU ARM + 12 GB RAM (o limite foi cortado pela metade em 2026), 200 GB disco, 10 TB/mês de saída. Instância ociosa pode ser recuperada pela Oracle. Exige cadastro com cartão. Não serve para Jev-Omni (sem GPU). | https://docs.oracle.com/en-us/iaas/Content/FreeTier/resourceref.htm |
| Google Cloud Run (CPU) | RF-DETR | 180.000 vCPU-s, 360.000 GiB-s e 2 milhões de requisições por mês grátis; GPU não tem cota grátis. Exige conta de faturamento. | https://cloud.google.com/run/pricing |

Não recomendo Colab nem Kaggle como servidor: são sessões de notebook, não serviços sempre acessíveis, e a T4 grátis (16 GB) não comporta o Jev-Omni em BF16.

Atenção: segundo a documentação atual, Spaces Gradio/Docker em CPU Basic exigem plano pago para serem criados; a exceção grátis é justamente o ZeroGPU.

### Pagas (sem contratar nada)
| Opção | GPU adequada ao Jev-Omni | Preço | Observação |
|---|---|---|---|
| Hugging Face PRO | ZeroGPU com 40 min/dia | US$ 9/mês; +US$ 1 por 10 min extra | Mais simples se a cota grátis não bastar |
| Modal | L40S 48 GB / A100 40 GB | US$ 1,95/h / US$ 2,10/h (cobrança por segundo); plano Starter inclui US$ 30/mês de crédito | Escala a zero; tempo de cold start para carregar 24 GB não medido. https://modal.com/pricing |
| HF Spaces dedicado | L40S 48 GB | US$ 1,80/h | Cobra enquanto ligado |
| HF Inference Endpoints | L40S 48 GB / A100 80 GB | US$ 1,80/h / US$ 2,50/h | https://huggingface.co/pricing |
| Google Cloud Run GPU | L4 24 GB (provavelmente insuficiente) | ~US$ 0,67/h | Preço da RTX Pro 6000 não verificado |

Para o RF-DETR, qualquer CPU pequena paga (ex.: HF CPU Upgrade US$ 0,03/h) já bastaria.

## 6. Recomendação

**Hugging Face ZeroGPU, com dois Spaces separados na sua conta: um para o Jev-Omni e outro para o RF-DETR.** É a única opção grátis que encontrei com GPU capaz de carregar o Jev-Omni, fica acessível pela internet com seu PC desligado, e o limite de 2 Spaces grátis cobre exatamente os dois ambientes. Limitações: cota de poucos minutos de GPU por dia, fila, e o Space dorme sem uso (a primeira chamada depois disso demora).

Alternativa para o RF-DETR se você quiser um servidor sempre ligado: Oracle Always Free. Alternativa paga para o Jev-Omni se a cota grátis não bastar: HF PRO (US$ 9/mês) ou Modal (créditos de US$ 30/mês).

## 7. Plano de testes reproduzível (próxima etapa, depende da sua aprovação)

Pré-requisitos: conta Hugging Face com e-mail verificado e mais de 30 dias; token de leitura (https://huggingface.co/settings/tokens).

**Jev-Omni**
1. Rodar `chamar_space_publico.py` contra o Space do autor com `astronaut.jpg`, perguntas "Há uma pessoa?" (esperado: Sim) e "Há um gato?" (esperado: Não). Registrar resposta, latência e cota consumida.
2. Criar o Space ZeroGPU próprio (cópia do app oficial, duração por chamada reduzida) e repetir o passo 1 com o PC desligado, chamando de outro dispositivo (ex.: celular ou Colab).

**RF-DETR**
3. Criar o Space ZeroGPU próprio com `rfdetr` (Gradio) e enviar as três imagens de teste. Critério de sucesso: mesmas classes da tabela da seção 3 (person; cat; cup, spoon, dining table) com confiança semelhante.
4. Repetir com o PC desligado, de outro dispositivo; registrar latência com o Space acordado e "frio".

Fora do escopo desta etapa: vídeo ao vivo e processamento em lote.

## 7.1 Verificação do Space do Jev-Omni (14/10/2026, sem execução)
Verificado por leitura de páginas públicas, porque o container bloqueia conexões diretas a huggingface.co e *.hf.space (proxy 403):
- `https://huggingface.co/api/spaces/akhilaaa3/jev-omni`: id `akhilaaa3/jev-omni`, sdk gradio, estágio RUNNING, hardware `zero-a10g`, público, última modificação 2026-09-22, sha 316c44f.
- `https://akhilaaa3-jev-omni.hf.space/gradio_api/info`: único endpoint `/decide`, parâmetros na ordem `state`, `question`, `options_text` (padrão "Yes\nNo"), `media` (arquivo), `modality` (padrão "text"); retorna um Label com probabilidades e um texto Markdown. Isso bate com `chamar_space_publico.py`.
- O hardware informado (`zero-a10g`) difere da GPU citada na documentação atual do ZeroGPU (RTX Pro 6000); não sei qual GPU atende de fato.
- `astronaut.jpg` (foto de exemplo do scikit-image) foi copiada para `exemplos/jev-omni/`.
- **`client.predict()` não foi executado. Não há resultado do Jev-Omni.**

### Teste manual executado pelo Daniel (14/10/2026, ~14:03 UTC, pela interface web, logado)
Space akhilaaa3/jev-omni, modality image, `astronaut.jpg` (44,9 KB), Situation "A photo sent by a user", Question "Is there a person in the image?", opções Yes/No.
Resultado exibido: **Yes 100%, No 0%**, latência informada pelo app **1049 ms**. Evidência: print enviado na thread.
Isso comprova que o Jev-Omni roda remotamente (servidor de terceiro, ZeroGPU) e responde a uma imagem enviada pelo navegador. Ainda não testado: chamada por script (`gradio_client`), teste negativo, e um Space próprio.

### Teste por script executado pelo Daniel (14/10/2026, ~14:31 UTC, Windows, Python 3.13.16, gradio_client 2.7.2)
`py -3.13 chamar_space_publico.py` com `astronaut.jpg`, pergunta "Há uma pessoa na imagem?", opções Sim/Não, token HF de leitura.
- `view_api()` confirmou: 1 endpoint `/decide`, parâmetros `state, question, options_text, media, modality`.
- Saída: `{'label': 'Sim', 'confidences': [{'label': 'Sim', 'confidence': 0.99966}, {'label': 'Não', 'confidence': 0.00034}]}` e `**Sim** · 100.0% · 1056 ms`.
- Observação: no Python 3.15 a instalação falhou (pyyaml sem wheel para 3.15 e compilação bloqueada pela política de Controle de Aplicativo do Windows). Usar Python 3.13.

### Teste negativo por script (Daniel, 14/10/2026 ~15:44 UTC)
`py -3.13 teste_negativo.py`: mesma `astronaut.jpg`, pergunta "Há um gato na imagem?", opções Sim/Não.
Saída: `{'label': 'Não', 'confidences': [{'label': 'Não', 'confidence': 0.99991}, {'label': 'Sim', 'confidence': 8.75e-05}]}` e `**Não** · 100.0% · 1035 ms`. Resultado esperado (Não) confirmado.

Conclusão: envio de imagem por script para o Jev-Omni remoto e recebimento do resultado **comprovados**, usando o Space de terceiro.

Como verificar manualmente: abra https://huggingface.co/spaces/akhilaaa3/jev-omni, escolha "image", envie a foto, preencha Situation, Question e Options e clique em Decide; ou rode `chamar_space_publico.py` no seu PC e envie a saída para registro aqui.

## 7.2 Spaces preparados (14/10/2026)
Arquivos em `spaces/`, com guia em `spaces/GUIA.md`. O app Gradio do RF-DETR (`spaces/rf-detr/app.py`) foi testado localmente: chamado por `gradio_client`, devolveu cup 0,941, spoon 0,903, dining table 0,87. Não testado dentro do Hugging Face. Jev-Omni: duplicar o Space do autor.

## 8. Limitações desta pesquisa
- Este container não tem GPU e bloqueia huggingface.co e download.pytorch.org, por isso o Jev-Omni não foi executado e o torch foi instalado pelo PyPI.
- Benchmarks do Jev-Omni são do autor. Preços podem mudar; a regra exata de consumo de cota do ZeroGPU não está documentada oficialmente.
