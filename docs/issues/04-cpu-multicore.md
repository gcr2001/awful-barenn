# Issue #4 - CPU v3: multicore onde o paralelismo é exato
**Título:** `CPU v3: leitura e avaliação paralelas (OpenMP) e por que o treino não pode ser paralelizado`
**Branch:** `perf/cpu-multicore` · **Tag de fechamento:** `v0.4-cpu-multicore` · **PR:** "perf: leitura e avaliação paralelas com OpenMP"
---
## ABERTURA (colar ao abrir a Issue, antes de qualquer commit)
## Diagnóstico:  Qual gargalo você encontrou?
Com o treino em 0,037 s, a leitura (strtof/istream, 22,3 ms) e a avaliação passam a pesar. O treino é SGD amostra-a-amostra: cada passo depende dos pesos do anterior (Amdahl: fração serial dominante).
## Hipótese:  Qual conceito técnico você pretende aplicar para resolver esse gargalo?
Paralelizar só o que é exato: parsing do arquivo (`leitura_rapida.hpp`, testado em `tests/test_leitura.cpp`, 25 casos de equivalência) e avaliação (somas por amostra independentes). OpenMP dentro da amostra foi testado e rejeitado (`experimentos/`).
## Métrica base: Qual é o tempo ou custo de processamento atual dessa rotina?
| Ambiente de Teste | Arquivo de entrada | Tempo/N Chamadas/outra métricas |
|:---:|:---:|:---:|
| compute10 | optdigits.tra | carga 22,3 ms; treino 0,0367 s |
| compute20 | x100 (382.300 amostras) | leitura legado 2220,7 ms |
## FECHAMENTO (colar como comentário ao fechar, com a tag)
Tag `v0.4-cpu-multicore`. Carga 22,3 → 0,6 ms; wall 0,380 → 0,041 s (9,27×). x100: leitura 2220,7 → 36,2 ms (8 threads). OpenMP dentro da amostra: 12× a 39× mais lento em três rodadas (região vazia 0,42–12,0 µs vs 0,19 µs de trabalho). Hash idêntico.
