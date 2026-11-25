#!/bin/bash
# Mede o legado, as versoes de CPU e as versoes CUDA no no de GPU do Franky,
# todas na mesma alocacao (o clock da GPU e a variacao entre nos mudam ~10%).
#
# Uso (a partir da raiz do repositorio, no login node):
#   make                      # compila CPU e CUDA no login node
#   sbatch slurm/gpu.slurm    # REP=5 sbatch slurm/gpu.slurm para mudar as repeticoes
#SBATCH --job-name=barenn-gpu
#SBATCH --partition=sunny_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=barenn-gpu-%j.out
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
source slurm/ambiente.sh

BINS=()
for b in legado cpu_v1 cpu_v2 cpu_v3 cuda_v1 cuda_v2 cuda_v3 cuda_v4; do [ -x "bin/$b" ] && BINS+=("bin/$b"); done
[ ${#BINS[@]} -gt 0 ] || { echo "nenhum binario em bin/: rode 'make' no login node antes"; exit 1; }

scripts/comparar.sh "${REP:-9}" "${BINS[@]}"
