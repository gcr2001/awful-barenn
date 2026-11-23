# Issue #2 - CPU v1: eliminar trabalho morto, cópias e alocações
**Título:** `CPU v1: remover grad_in morto da camada 1 e cópias/alocações por amostra`
**Branch:** `perf/cpu-trabalho-morto` · **Tag de fechamento:** `v0.2-cpu-trabalho-morto` · **PR:** "perf: remove grad_in morto da camada 1 e alocações por amostra"
---
## ABERTURA (colar ao abrir a Issue, antes de qualquer commit)
## Diagnóstico:  Qual gargalo você encontrou?
Do perfil da Issue #1: o grad_in da camada 1 é calculado e descartado (75 % do laço de grad_in) e há ~3 % em `malloc`/`free`/`operator new` por amostra, mais cópias de vetores.
## Hipótese:  Qual conceito técnico você pretende aplicar para resolver esse gargalo?
Eliminar o cálculo morto e usar buffers pré-alocados (`common.hpp`, `cpu_v1.cpp`). Não muda nenhuma soma, então o hash deve permanecer idêntico. Esperado: redução de instruções próxima à soma dos dois itens (~14 %+ do total).
## Métrica base: Qual é o tempo ou custo de processamento atual dessa rotina?
| Ambiente de Teste | Arquivo de entrada | Tempo/N Chamadas/outra métricas |
|:---:|:---:|:---:|
| compute10, i7-13700 | optdigits.tra | legado -O3: treino 0,3543 s; 9,728 G Ir |
## FECHAMENTO (colar como comentário ao fechar, com a tag)
Tag `v0.2-cpu-trabalho-morto`. Treino 0,3543 → 0,1693 s (2,09×), Ir 14,03 G → 3,35 G (callgrind). Hash idêntico.
