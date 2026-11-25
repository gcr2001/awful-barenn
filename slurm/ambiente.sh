#!/usr/bin/env bash
# Registra no topo da saida do job tudo o que e preciso para reproduzir a
# medicao: versao do codigo, no, CPU, GPU, compiladores e flags.
# Uso: source slurm/ambiente.sh   (chamado pelos jobs em slurm/*.slurm)

echo "################ AMBIENTE ################"
echo "job      : ${SLURM_JOB_ID:-sem-slurm}  particao=${SLURM_JOB_PARTITION:-?}  cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "no       : $(hostname)"
echo "data     : $(date '+%F %T')"
echo "codigo   : $(git describe --tags --always --dirty 2>/dev/null || echo 'fora de um repositorio git') (commit $(git rev-parse --short HEAD 2>/dev/null || echo '?'))"
if command -v lscpu >/dev/null 2>&1; then
  lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|L1d|L2|L3)' | sed 's/^/cpu      : /'
fi
echo "OMP      : OMP_NUM_THREADS=${OMP_NUM_THREADS:-<padrao>}"
if command -v g++ >/dev/null 2>&1; then echo "g++      : $(g++ --version | head -1)"; fi
NVCC_BIN="${NVCC:-/usr/local/cuda/bin/nvcc}"
if [ -x "$NVCC_BIN" ]; then echo "nvcc     : $("$NVCC_BIN" --version | tail -1)"; fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,clocks.sm,clocks.max.sm,driver_version,temperature.gpu \
             --format=csv,noheader 2>/dev/null | sed 's/^/gpu      : /'
fi
# Flags de compilacao realmente usadas (o Makefile e a fonte da verdade).
grep -E '^(STD|ISA|ARCH|FP)[[:space:]]*[:?]?=' Makefile 2>/dev/null | sed 's/^/flags    : /'
echo "##########################################"
