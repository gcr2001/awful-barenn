# Issue #6 - CUDA v2: um kernel fundido por amostra
**Título:** `CUDA v2: fundir os 10 kernels do passo de SGD em um só (memória compartilhada + __syncthreads)`
**Branch:** `perf/cuda-kernel-fundido` · **Tag de fechamento:** `v0.6-cuda-kernel-fundido` · **PR:** "perf: funde os 10 kernels do passo em um"
---
## ABERTURA (colar ao abrir a Issue, antes de qualquer commit)
## Diagnóstico:  Qual gargalo você encontrou?
nsys da v1: o tempo é dominado por lançamento e gaps entre kernels (piso de ~1,24 µs por kernel × 10 por amostra).
## Hipótese:  Qual conceito técnico você pretende aplicar para resolver esse gargalo?
Um único kernel por amostra: ativações/deltas em memória compartilhada, `__syncthreads` entre fases (`cuda_passo.cuh`, `cuda_v2.cu`).
## Métrica base: Qual é o tempo ou custo de processamento atual dessa rotina?
| Ambiente de Teste | Arquivo de entrada | Tempo/N Chamadas/outra métricas |
|:---:|:---:|:---:|
| compute20, RTX A1000 | optdigits.tra | v1: treino 3,6614 s |
## FECHAMENTO (colar como comentário ao fechar, com a tag)
Tag `v0.6-cuda-kernel-fundido`. Treino 3,6614 → 0,9687 s (3,8×). Hash idêntico. Resta 1 lançamento por amostra.
