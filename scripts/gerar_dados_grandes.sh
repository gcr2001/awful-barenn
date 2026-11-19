#!/usr/bin/env bash
# Gera datasets maiores replicando as linhas de optdigits.tra/.tes K vezes.
# Serve para estudar escalabilidade quando o volume de dados cresce (o
# conteudo e o mesmo, so o tamanho muda; o tempo de treino cresce
# linearmente com o numero de amostras).
#
# Uso: scripts/gerar_dados_grandes.sh <K> [destino=dados]
# Gera: <destino>/x<K>.tra e <destino>/x<K>.tes
set -euo pipefail
K="${1:?uso: gerar_dados_grandes.sh <K> [destino]}"
DEST="${2:-dados}"
cd "$(dirname "$0")/.."
mkdir -p "$DEST"
for ext in tra tes; do
  : > "$DEST/x$K.$ext"
  for _ in $(seq 1 "$K"); do cat "optdigits.$ext" >> "$DEST/x$K.$ext"; done
done
wc -l "$DEST/x$K.tra" "$DEST/x$K.tes"
