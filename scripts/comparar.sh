#!/usr/bin/env bash
# Mede várias versões DENTRO DA MESMA ALOCAÇÃO (mesmo nó, mesmo estado de
# clock/temperatura), uma após a outra. Necessário porque nós diferentes do
# cluster podem diferir ~10% em tempo; só assim a comparação entre versões é
# justa.
#
# Uso (dentro de um srun/sbatch):
#   scripts/comparar.sh <repeticoes> <bin1> [bin2 ...]
# Variaveis opcionais: BENCH_ARGS="treino.csv teste.csv"  BENCH_REF_DIR=ref/x100
# Saída: uma linha resumo por binário (treino_dp = desvio padrão amostral do
# treino) + hostname e (se houver) a GPU.
set -u
REP="${1:?uso: comparar.sh <repeticoes> <bin>...}"
shift
echo "# no: $(hostname)  data: $(date +%F\ %T)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu --format=csv,noheader 2>/dev/null | sed 's/^/# gpu: /'
fi
printf "%-22s %10s %10s %10s %10s %10s  %s\n" "binario" "treino_min" "treino_med" "treino_dp" "wall_med" "carga_med" "corretude"
for bin in "$@"; do
  saida=$(scripts/bench.sh "$bin" "$REP" ${BENCH_ARGS:-} 2>&1)
  t_min=$(echo "$saida" | awk '/^Tempo de treino/{sub("min=","",$5); print $5}')
  t_med=$(echo "$saida" | awk '/^Tempo de treino/{sub("mediana=","",$6); print $6}')
  t_dp=$(echo "$saida"  | awk '/^Tempo de treino/{for(i=1;i<=NF;i++) if($i ~ /^desvio=/){sub("desvio=","",$i); print $i}}')
  w_med=$(echo "$saida" | awk '/^Tempo total/{sub("mediana=","",$5); print $5}')
  c_med=$(echo "$saida" | awk '/^Tempo de carga \(s\)  *min=/{sub("mediana=","",$6); print $6}')
  ok_txt=$(echo "$saida" | grep -c "Saida textual : OK")
  ok_hash=$(echo "$saida" | grep -c "Hash bit a bit: OK")
  if [ "$ok_txt" -eq 1 ] && [ "$ok_hash" -eq 1 ]; then c="OK(texto+hash)"; \
  elif [ "$ok_txt" -eq 1 ]; then c="OK(texto)"; else c="DIVERGE"; fi
  printf "%-22s %10s %10s %10s %10s %10s  %s\n" "$(basename "$bin")" "${t_min:--}" "${t_med:--}" "${t_dp:--}" "${w_med:--}" "${c_med:--}" "$c"
done
exit 0
