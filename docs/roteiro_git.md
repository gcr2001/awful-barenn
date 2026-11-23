# Roteiro Git (ordem exata)

Branch padrão do repositório: **`master`** (o README fala em `main`; use `master`).

## 0. Antes de tudo: abrir as 8 Issues (#1 a #8)

Issues e PRs dividem o mesmo contador. Abra as 8 Issues **antes** de qualquer PR, para que sejam #1–#8 e os links do README funcionem. O corpo de cada uma é a seção `ABERTURA` de `docs/issues/NN-*.md` (o título está no topo do arquivo). A seção `FECHAMENTO` é comentada ao fechar.

Limpeza: `callgrind.out.45358` está versionado desde o começo e é lixo; remova no bloco 1 (`git rm callgrind.out.45358`).

## 1. Por bloco

```bash
git switch master && git pull
git switch -c <branch>
git add <arquivos>
git commit -m "<tipo>: <mensagem>"
git push -u origin <branch>
# PR com "Closes #N", merge em master
git switch master && git pull
git tag -a <tag> -m "<mensagem>"
git push origin <tag>
# comentar FECHAMENTO na Issue #N citando a tag e fechar
```

Se quiser mais de um commit por bloco (recomendado, "commits atômicos"), separe, p.ex., código e teste/experimento.

| # | Branch | Tag | Arquivos | Commit sugerido |
|---|---|---|---|---|
| 1 | `test/baseline-harness` | `v0.1-baseline` | `Makefile .gitignore scripts/bench.sh scripts/instrumentar_baseline.sh scripts/gerar_referencia.sh ref/saida_referencia.txt ref/hashes_referencia.txt` (+ `git rm callgrind.out.45358`) | `test: harness de benchmark, referência bit a bit e perfil do legado` |
| 2 | `perf/cpu-trabalho-morto` | `v0.2-cpu-trabalho-morto` | `src/common.hpp src/cpu_v1.cpp` | `perf: remove grad_in morto da camada 1 e alocações por amostra` |
| 3 | `perf/cpu-simd` | `v0.3-cpu-simd` | `src/rede_simd.hpp src/cpu_v2.cpp` | `perf: vetoriza entre neurônios preservando a ordem da soma` |
| 4 | `perf/cpu-multicore` | `v0.4-cpu-multicore` | `src/leitura_rapida.hpp src/cpu_v3.cpp tests/test_leitura.cpp scripts/gerar_dados_grandes.sh ref/x100/* experimentos/omp_forkjoin.cpp experimentos/cpu_omp_amostra.cpp` | `perf: leitura e avaliação paralelas com OpenMP` |
| 5 | `feat/cuda-port` | `v0.5-cuda-port` | `src/cuda_util.cuh src/cuda_v1.cu scripts/comparar.sh` | `feat: port fiel do treino para CUDA` |
| 6 | `perf/cuda-kernel-fundido` | `v0.6-cuda-kernel-fundido` | `src/cuda_passo.cuh src/cuda_v2.cu experimentos/gerar_cuda_v1_graph.py` | `perf: funde os 10 kernels do passo em um` |
| 7 | `perf/cuda-persistente` | `v0.7-cuda-persistente` | `src/cuda_v3.cu experimentos/cuda_fases.cu` | `perf: kernel persistente com pesos em shared e prefetch` |
| 8 | `perf/cuda-latencia` | `v0.8-cuda-latencia` | `src/cuda_avaliar.cuh src/cuda_v4.cu experimentos/gerar_fases_v4.py` | `perf: esconde latência de shared no passo de SGD` |
| 9 | `docs/readme-final` | `v1.0-entrega` | `README.md docs/**` | `docs: resumo executivo, resultados e evidências` |

Os blocos 1–8 fecham as Issues #1–#8, respectivamente. O bloco 9 não tem Issue própria.

## 2. Mensagens sugeridas para tags anotadas

- `v0.1-baseline`: "Baseline reprodutivel do legado: treino 0.354 s (-O3, i7-13700); harness, referencia bit a bit e perfil callgrind. Sem alteracao do codigo."
- `v0.2-cpu-trabalho-morto`: "CPU v1: remove grad_in morto e alocacoes; treino 0.354 -> 0.169 s (2.09x), 14.0 G -> 3.35 G Ir; hash identico."
- `v0.3-cpu-simd`: "CPU v2: SIMD entre neuronios sem reassociar soma; treino 0.0367 s (9.65x sobre legado), 1.315 G Ir; hash identico."
- `v0.4-cpu-multicore`: "CPU v3: leitura/avaliacao paralelas; carga 22.3 -> 0.6 ms, wall 0.380 -> 0.041 s (9.27x); treino segue serial (SGD); hash identico."
- `v0.5-cuda-port`: "CUDA v1: port fiel, hash identico; treino 3.66 s (11.9x mais lento que o legado), 10 kernels/amostra, launch-bound (nsys)."
- `v0.6-cuda-kernel-fundido`: "CUDA v2: kernel fundido; treino 3.66 -> 0.969 s (3.8x); hash identico."
- `v0.7-cuda-persistente`: "CUDA v3: kernel persistente, pesos em shared; treino 0.969 -> 0.443 s; hash identico."
- `v0.8-cuda-latencia`: "CUDA v4: latencia de shared escondida; treino 0.443 -> 0.280 s (1.10x mais rapido que o legado, 7.2x mais lento que CPU v2); 5232 -> 2391 ciclos/amostra; hash identico."
- `v1.0-entrega`: "Entrega final: resumo executivo, resultados e evidencias."

## 3. Verificação de reprodutibilidade das tags

`scripts/verificar_blocos.sh` simula os 8 blocos cumulativos (cada um compila sozinho e todos os binários passam no hash). Rodado no cluster com:

```bash
RUN_CMD="srun --partition=sunny_gpu --gres=gpu:1 --mem=2G --ntasks=1" scripts/verificar_blocos.sh
```

Após taguear, em qualquer tag: `git checkout <tag> && make` (compile no nó de login) e rode com `srun`; veja `docs/RESULTADOS.md`.

## 4. Fechamento

Todos os branches mesclados em `master`, Issues #1–#8 fechadas com o texto `FECHAMENTO` (que cita a tag). Os links do README assumem `https://github.com/gcr2001/awful-barenn/issues/N` e `.../releases/tag/<tag>`.
