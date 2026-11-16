# Issue #8 - CUDA v4: esconder a latência dentro do passo de SGD
**Título:** `CUDA v4: reescrever o passo para esconder latência de memória compartilhada (registradores, papéis por warp, menos barreiras)`
**Branch:** `perf/cuda-latencia` · **Tag de fechamento:** `v0.8-cuda-latencia` · **PR:** "perf: esconde latência de shared no passo de SGD"
---
## ABERTURA (colar ao abrir a Issue, antes de qualquer commit)
## Diagnóstico:  Qual gargalo você encontrou?
Fases medidas por clock64 na v3: 5232 ciclos/amostra, dominados por latência de shared e barreiras em cadeia, não por FLOP (≈0,15 % do pico).
## Hipótese:  Qual conceito técnico você pretende aplicar para resolver esse gargalo?
Reorganizar o passo: dados em registradores, papéis por warp, menos `__syncthreads`, `__launch_bounds__(T,1)`, padrão carregar-tudo/calcular/armazenar com barreira de compilador (`cuda_avaliar.cuh`, `cuda_v4.cu`).
## Métrica base: Qual é o tempo ou custo de processamento atual dessa rotina?
| Ambiente de Teste | Arquivo de entrada | Tempo/N Chamadas/outra métricas |
|:---:|:---:|:---:|
| compute20, RTX A1000 | optdigits.tra | v3: treino 0,4411–0,4434 s; 5232 ciclos/amostra |
## FECHAMENTO (colar como comentário ao fechar, com a tag)
Tag `v0.8-cuda-latencia`. Treino 0,4434 → 0,2798 s (1,10× mais rápido que o legado, 7,2× mais lento que a CPU v2); ablação 0,4411 → 0,3225 → 0,3164 → 0,2781; 2391 ciclos/amostra; 92 regs, 0 spill; TH=128 ótimo. Hash idêntico. x100: 24,40 s vs legado 30,57 s e CPU v3 3,81 s.
