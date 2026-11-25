#!/bin/bash
# Mede o legado e as versoes de CPU no no de CPU do Franky, na mesma alocacao.
#
# Uso (a partir da raiz do repositorio, no login node):
#   make                      # compila no login node: os nos de computacao nao tem headers
#   sbatch slurm/cpu.slurm    # REP=5 sbatch slurm/cpu.slurm para mudar as repeticoes
#SBATCH --job-name=barenn-cpu
#SBATCH --partition=sunny_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=barenn-cpu-%j.out
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
source slurm/ambiente.sh

BINS=()
for b in legado legado_v3 cpu_v1 cpu_v2 cpu_v3; do [ -x "bin/$b" ] && BINS+=("bin/$b"); done
[ ${#BINS[@]} -gt 0 ] || { echo "nenhum binario em bin/: rode 'make' no login node antes"; exit 1; }

scripts/comparar.sh "${REP:-9}" "${BINS[@]}"
