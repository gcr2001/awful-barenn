# Issue #7 - CUDA v3: kernel persistente, pesos em shared, prefetch e avaliação na GPU
**Título:** `CUDA v3: kernel persistente + pesos em memória compartilhada + prefetch + avaliação em stream separada`
**Branch:** `perf/cuda-persistente` · **Tag de fechamento:** `v0.7-cuda-persistente` · **PR:** "perf: kernel persistente com pesos em shared e prefetch"
---
## ABERTURA (colar ao abrir a Issue, antes de qualquer commit)
## Diagnóstico:  Qual gargalo você encontrou?
Na v2 ainda há 1 lançamento por amostra e os pesos são relidos da memória global a cada passo.
## Hipótese:  Qual conceito técnico você pretende aplicar para resolver esse gargalo?
Kernel persistente (um bloco percorre todas as amostras/épocas), pesos residentes em shared, prefetch da próxima amostra, avaliação em kernel/stream próprio (`cuda_v3.cu`; fases medidas com `experimentos/cuda_fases.cu`).
## Métrica base: Qual é o tempo ou custo de processamento atual dessa rotina?
| Ambiente de Teste | Arquivo de entrada | Tempo/N Chamadas/outra métricas |
|:---:|:---:|:---:|
| compute20, RTX A1000 | optdigits.tra | v2: treino 0,9687 s |
## FECHAMENTO (colar como comentário ao fechar, com a tag)
Tag `v0.7-cuda-persistente`. Treino 0,9687 → 0,4434 s. clock64: 5232 ciclos/amostra. Hash idêntico.
