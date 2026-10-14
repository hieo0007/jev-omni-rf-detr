# Guia de implantação dos dois Spaces (usar a partir de ~13/11/2026)

Pré-requisito: conta Hugging Face com mais de 30 dias e e-mail verificado (limite grátis: 2 Spaces ZeroGPU).
Cada modelo fica no seu próprio Space, separado do outro.

## Space 1: RF-DETR
1. Em https://huggingface.co/new-space: nome `rf-detr`, SDK **Gradio**, hardware **ZeroGPU**, público.
2. Em "Files", envie `app.py`, `requirements.txt` e `README.md` da pasta `rf-detr/`.
3. Espere o status ficar "Running" (a primeira subida baixa ~349 MB de pesos).
4. Teste no PC: `py -3.13 enviar_imagem.py SEU_USUARIO/rf-detr coffee.jpg`
   Esperado (igual ao teste local): cup ~0,94, spoon ~0,90, dining table ~0,87.

Testado em 14/10/2026 só localmente (servidor Gradio em 127.0.0.1, chamado via gradio_client): resultado acima confirmado.
Ainda não testado dentro do Hugging Face.

## Space 2: Jev-Omni
O caminho mais simples é duplicar o Space oficial do autor, que já funciona (testado em 14/10/2026):
1. Abra https://huggingface.co/spaces/akhilaaa3/jev-omni, menu "⋮" > **Duplicate this Space**.
2. Mantenha hardware ZeroGPU e visibilidade pública.
3. Teste com `chamar_space_publico.py`, trocando `"akhilaaa3/jev-omni"` por `"SEU_USUARIO/jev-omni"`.

## Teste final (objetivo do projeto)
Com o PC desligado, chame os dois Spaces de outro dispositivo (celular pelo navegador, ou Colab com gradio_client).
Registre resultado e latência com o Space acordado e depois de ele dormir.

## Cotas
Cada chamada consome a cota diária de GPU de quem chama (5 min/dia em conta grátis). O RF-DETR reserva 10 s por chamada; o Jev-Omni do autor reserva 45 s.
