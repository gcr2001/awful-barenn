#!/usr/bin/env bash
# Gera a saida de referencia (ref/) a partir do codigo LEGADO, compilado de
# forma "pura" (x86-64 sem FMA, sem fast-math): e o gabarito de corretude.
#
# Compilar exige os headers do sistema (login node); a execucao pode ir para
# um no de computacao via RUN_CMD.
# Uso (na raiz do repo):
#   RUN_CMD="srun --partition=sunny_cpu --mem=1G" scripts/gerar_referencia.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build ref
scripts/instrumentar_baseline.sh build/barenn_ref_hash.cpp
RPATH="$(dirname "$(g++ -print-file-name=libstdc++.so)")"
g++ -std=c++17 -O3 -ffp-contract=off -Wl,-rpath,"$RPATH" build/barenn_ref_hash.cpp -o build/barenn_ref_hash
BARENN_HASH=1 ${RUN_CMD:-} ./build/barenn_ref_hash > build/ref_completa.txt
grep -v -E '^(Tempo|HASH|INFO)' build/ref_completa.txt > ref/saida_referencia.txt
grep -E '^HASH_' build/ref_completa.txt > ref/hashes_referencia.txt
echo "--- ref/saida_referencia.txt ---";  cat ref/saida_referencia.txt
echo "--- ref/hashes_referencia.txt ---"; cat ref/hashes_referencia.txt
