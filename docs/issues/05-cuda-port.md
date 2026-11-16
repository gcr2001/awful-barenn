# Issue #5 - CUDA v1: port fiel para a GPU
**Título:** `CUDA v1: port fiel do treino (um thread por neurônio) com resultado idêntico ao legado`
**Branch:** `feat/cuda-port` · **Tag de fechamento:** `v0.5-cuda-port` · **PR:** "feat: port fiel do treino para CUDA"
---
## ABERTURA (colar ao abrir a Issue, antes de qualquer commit)
## Diagnóstico:  Qual gargalo você encontrou?
Primeira versão GPU: o passo de SGD vira 10 kernels por amostra (forward/backward por camada, update). O problema é minúsculo (2.720 pesos, ~15 k FLOP/amostra).
## Hipótese:  Qual conceito técnico você pretende aplicar para resolver esse gargalo?
Port direto, um thread por neurônio, `--fmad=false` para manter o hash. Espera-se resultado correto e desempenho ruim, estabelecendo a linha de base CUDA e o diagnóstico com nsys (`cuda_util.cuh`, `cuda_v1.cu`, `comparar.sh`).
## Métrica base: Qual é o tempo ou custo de processamento atual dessa rotina?
| Ambiente de Teste | Arquivo de entrada | Tempo/N Chamadas/outra métricas |
|:---:|:---:|:---:|
| compute20, RTX A1000 | optdigits.tra | legado no nó: treino 0,3088 s |
| nsys 1 época | optdigits.tra | cudaLaunchKernel 38.230 chamadas, média 2.567 ns; kernels 1,2–2,9 µs; GPU ocupada ~72 % |
## FECHAMENTO (colar como comentário ao fechar, com a tag)
Tag `v0.5-cuda-port`. Hash idêntico. Treino 3,6614 s (11,9× MAIS LENTO que o legado). Causa (nsys): custo de lançamento/latência por kernel. CUDA Graph mitiga só 1,37× (experimento).
