#!/bin/bash
# Teste de escala: dataset 100x maior (382.300 amostras de treino e 179.700 de
# teste), conferido contra ref/x100. Roda no no de GPU para comparar CPU e CUDA
# na mesma alocacao.
#
# Uso (a partir da raiz do repositorio, no login node):
#   make
#   sbatch slurm/x100.slurm   # gera dados/x100.* se faltar; REP=3 por padrao (legado ~32 s cada)
#SBATCH --job-name=barenn-x100
#SBATCH --partition=sunny_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=barenn-x100-%j.out
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
source slurm/ambiente.sh

[ -f dados/x100.tra ] && [ -f dados/x100.tes ] || scripts/gerar_dados_grandes.sh 100

BINS=()
for b in legado cpu_v2 cpu_v3 cuda_v3 cuda_v4; do [ -x "bin/$b" ] && BINS+=("bin/$b"); done
[ ${#BINS[@]} -gt 0 ] || { echo "nenhum binario em bin/: rode 'make' no login node antes"; exit 1; }

BENCH_ARGS="dados/x100.tra dados/x100.tes" BENCH_REF_DIR=ref/x100 scripts/comparar.sh "${REP:-3}" "${BINS[@]}"
