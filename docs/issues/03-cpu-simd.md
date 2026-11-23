# Issue #3 - CPU v2: layout SIMD vetorizando entre neurônios
**Título:** `CPU v2: vetorizar entre neurônios (layout [entrada][saída]) sem reassociar a soma`
**Branch:** `perf/cpu-simd` · **Tag de fechamento:** `v0.3-cpu-simd` · **PR:** "perf: vetoriza entre neurônios preservando a ordem da soma"
---
## ABERTURA (colar ao abrir a Issue, antes de qualquer commit)
## Diagnóstico:  Qual gargalo você encontrou?
Após a v1 o custo restante é escalar. O layout do legado `weights[saida][entrada]` faz o compilador querer vetorizar a redução sobre a entrada, o que reassocia a soma e muda o hash (e é proibido sem `-ffast-math`).
## Hipótese:  Qual conceito técnico você pretende aplicar para resolver esse gargalo?
Transpor para `[entrada][saída]` e vetorizar **entre neurônios**: cada lane soma a sua própria sequência na mesma ordem do legado. Bit-exato com `-ffp-contract=off` (`rede_simd.hpp`, `cpu_v2.cpp`).
## Métrica base: Qual é o tempo ou custo de processamento atual dessa rotina?
| Ambiente de Teste | Arquivo de entrada | Tempo/N Chamadas/outra métricas |
|:---:|:---:|:---:|
| compute10, i7-13700 | optdigits.tra | v1: treino 0,1693 s; 3,35 G Ir |
## FECHAMENTO (colar como comentário ao fechar, com a tag)
Tag `v0.3-cpu-simd`. Treino 0,0367 s (9,65× sobre o legado, 4,6× sobre v1), 1,315 G Ir; passo por amostra 42,7 k → 4,46 k instruções; ≈ 79 GFLOP/s. Hash idêntico.
