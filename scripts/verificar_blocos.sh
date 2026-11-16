#!/usr/bin/env bash
# Verifica que CADA bloco logico compila sozinho, ou seja, que `git checkout <tag>`
# + `make` funciona em todos os marcos. Para cada bloco, copia para um diretorio
# temporario SOMENTE os arquivos acumulados ate aquele bloco (a mesma lista do
# docs/roteiro_git.md) e executa `make`. Nao mexe no Git.
#
# Uso (na raiz do repo, num host com g++ e, opcionalmente, nvcc):
#   scripts/verificar_blocos.sh            # compila
#   RUN_CMD="srun -p sunny_cpu --mem=1G" scripts/verificar_blocos.sh   # + executa e confere o hash
set -u
cd "$(dirname "$0")/.."
RAIZ="$PWD"

# Arquivos ADICIONADOS por cada bloco (o legado e os dados ja existem no commit inicial).
BASE="barenn.cpp optdigits.tra optdigits.tes optdigits.names"
B0="Makefile .gitignore scripts/bench.sh scripts/instrumentar_baseline.sh scripts/gerar_referencia.sh ref/saida_referencia.txt ref/hashes_referencia.txt"
B1="src/common.hpp src/cpu_v1.cpp"
B2="src/rede_simd.hpp src/cpu_v2.cpp"
B3="src/leitura_rapida.hpp src/cpu_v3.cpp tests/test_leitura.cpp scripts/gerar_dados_grandes.sh ref/x100/saida_referencia.txt ref/x100/hashes_referencia.txt experimentos/omp_forkjoin.cpp experimentos/cpu_omp_amostra.cpp"
B4="src/cuda_util.cuh src/cuda_v1.cu scripts/comparar.sh"
B5="src/cuda_passo.cuh src/cuda_v2.cu experimentos/gerar_cuda_v1_graph.py"
B6="src/cuda_v3.cu experimentos/cuda_fases.cu"
B7="src/cuda_avaliar.cuh src/cuda_v4.cu experimentos/gerar_fases_v4.py"
BLOCOS=("B0:v0.1-baseline:$B0" "B1:v0.2-cpu-trabalho-morto:$B1" "B2:v0.3-cpu-simd:$B2" "B3:v0.4-cpu-multicore:$B3"
        "B4:v0.5-cuda-port:$B4" "B5:v0.6-cuda-kernel-fundido:$B5" "B6:v0.7-cuda-persistente:$B6" "B7:v0.8-cuda-latencia:$B7")

# Dentro do projeto (e nao em /tmp) para que os nos de computacao enxerguem os binarios.
TMP=$(mktemp -d "$RAIZ/.verif.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
ACUM="$BASE"
falhas=0
for entrada in "${BLOCOS[@]}"; do
  id="${entrada%%:*}"; resto="${entrada#*:}"; tag="${resto%%:*}"; arqs="${resto#*:}"
  ACUM="$ACUM $arqs"
  D="$TMP/$id"; mkdir -p "$D"
  faltando=""
  for f in $ACUM; do
    if [ -f "$RAIZ/$f" ]; then mkdir -p "$D/$(dirname "$f")"; cp "$RAIZ/$f" "$D/$f"; else faltando="$faltando $f"; fi
  done
  if [ -n "$faltando" ]; then echo "[$id $tag] FALTAM arquivos:$faltando"; falhas=$((falhas+1)); continue; fi
  if ( cd "$D" && make >/dev/null 2>"$D/make.err" ); then
    bins=$(cd "$D" && ls bin | tr '\n' ' ')
    msg="compila  (bin: $bins)"
    if [ -n "${RUN_CMD:-}" ] && [ -f "$D/ref/hashes_referencia.txt" ]; then
      for b in $(cd "$D" && ls bin | grep -E '^(cpu|cuda)_v'); do
        saida=$(cd "$D" && BARENN_HASH=1 $RUN_CMD ./bin/$b 2>/dev/null | grep '^HASH_')
        if [ -z "$saida" ]; then h="SEM-SAIDA"; elif [ "$saida" = "$(cat "$D/ref/hashes_referencia.txt")" ]; then h=OK; else h=DIVERGE; fi
        msg="$msg | $b:$h"
      done
    fi
    echo "[$id $tag] $msg"
  else
    echo "[$id $tag] FALHA ao compilar:"; head -5 "$D/make.err"; falhas=$((falhas+1))
  fi
done
[ $falhas -eq 0 ] && echo "TODOS OS BLOCOS COMPILAM SOZINHOS" || echo "$falhas bloco(s) com problema"
exit $falhas
