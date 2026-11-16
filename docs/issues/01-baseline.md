# Issue #1 - Baseline reprodutível do legado, política numérica e mapa de gargalos
**Título:** `Baseline reprodutível do legado, política numérica e identificação dos gargalos`
**Branch:** `test/baseline-harness` · **Tag de fechamento:** `v0.1-baseline` · **PR:** "test: harness de benchmark, referência bit a bit e perfil do legado"
---
## ABERTURA (colar ao abrir a Issue, antes de qualquer commit)
## Diagnóstico:  Qual gargalo você encontrou?
Sem documentação, o primeiro passo é medir. `valgrind --tool=callgrind --cache-sim=yes` no legado (`g++ -O2 -g`, 14,03 G instruções):

| Função / linha (barenn.cpp) | Instruções (Ir) | % |
|---|---|---|
| `DenseLayer::Backward` | 9,45 G | **67,4 %** |
| `main` (com `Forward` inlined) | 3,34 G | 23,8 % |
| `weights[i][j] -= lr * (delta[i] * last_input[j])` | 3,18 G | 22,7 % |
| `sum += delta[j] * weights[j][i]` (grad_in) | 2,12 G | 15,1 % |
| `sum += input[j] * weights[i][j]` (forward) | 1,61 G | 11,5 % |

Cache miss D1/LL = **0,0 %**: pesos (2.720 floats = 10,9 KB) cabem na L1; o custo é volume de instruções. `l1.Backward` devolve o `grad_in` da primeira camada, que `main` descarta: 75 % do laço de grad_in é trabalho morto. Além disso, `-O3 -march=native` (FMA) muda o hash dos pesos (`24d050fe…`) com a mesma acurácia impressa (95,5 %): comparar a saída impressa não prova equivalência.
## Hipótese:  Qual conceito técnico você pretende aplicar para resolver esse gargalo?
Este bloco **não altera `barenn.cpp`**. Cria o que torna os próximos blocos verificáveis: `scripts/bench.sh` (N repetições, min/mediana/média em nó exclusivo), hash FNV-1a dos pesos finais e perdas por época como critério de corretude bit a bit (`ref/`), `scripts/instrumentar_baseline.sh` e Makefile. Política: nenhuma otimização pode alterar o hash (sem FMA, sem `-ffast-math`, sem reassociar somas).
## Métrica base: Qual é o tempo ou custo de processamento atual dessa rotina?
| Ambiente de Teste | Arquivo de entrada | Tempo/N Chamadas/outra métricas |
|:---:|:---:|:---:|
| compute10, i7-13700, GCC 14.2 `-O0` | optdigits.tra | 5,1909 s |
| idem `-O2` | optdigits.tra | 0,4608 s |
| idem `-O3` | optdigits.tra | treino 0,3544 s / wall 0,3801 s |
| callgrind `-O2 -g` | optdigits.tra | 14,03 G Ir; `-O3 -g` 9,728 G Ir |
## FECHAMENTO (colar como comentário ao fechar, com a tag)
Fechada pela tag `v0.1-baseline`. Baseline: 0,354 s de treino (-O3). Hash de referência `366b333cd5264116` / `0b1dad0f07974985`. Gargalos mapeados (Backward 67 %, trabalho morto 75 % do grad_in, 0 % cache miss). Evidências em `docs/evidencias/`.
