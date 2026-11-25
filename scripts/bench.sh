#!/usr/bin/env bash
# Mede um binario do Barenn de forma reproduzivel e confere a corretude.
#
# Uso:   scripts/bench.sh <binario> [repeticoes=7] [args do binario...]
# Saida: min / mediana / media / max / desvio padrao do "Tempo de treino" e do
#        tempo total (wall), alem de OK/DIVERGE para (a) a saida textual e (b) os hashes bit a bit
#        em relacao a ref/ (gerada a partir do legado).
#
# Rodar a partir da raiz do repositorio (usa optdigits.tra/.tes do diretorio
# atual). Em cluster, chame este script DENTRO de um job (sbatch/srun).
set -u
BIN="${1:?uso: bench.sh <binario> [repeticoes] [args...]}"
REP="${2:-7}"
shift $(( $# >= 2 ? 2 : 1 ))
ARGS=("$@")
# Diretorio de referencia: ref/ (dataset padrao) ou outro (p.ex. ref/x100) via
# BENCH_REF_DIR quando o binario recebe argumentos de dataset customizado.
REF_DIR="${BENCH_REF_DIR:-ref}"
REF_SAIDA="$REF_DIR/saida_referencia.txt"
REF_HASH="$REF_DIR/hashes_referencia.txt"
# A comparacao so faz sentido com dataset padrao ou com referencia explicita.
COMPARAR=0
if [ $# -eq 0 ] || [ -n "${BENCH_REF_DIR:-}" ]; then COMPARAR=1; fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
: > "$tmp/treino"; : > "$tmp/wall"; : > "$tmp/carga"; : > "$tmp/aval"

for i in $(seq 1 "$REP"); do
  ini=$(date +%s.%N)
  BARENN_HASH=1 "$BIN" "${ARGS[@]}" > "$tmp/out.$i" 2>&1
  fim=$(date +%s.%N)
  awk -v a="$ini" -v b="$fim" 'BEGIN{printf "%.6f\n", b-a}' >> "$tmp/wall"
  awk '/^Tempo de treino/{print $4}'    "$tmp/out.$i" >> "$tmp/treino"
  awk '/^Tempo de carga/{print $4}'     "$tmp/out.$i" >> "$tmp/carga"
  awk '/^Tempo de avaliacao/{print $4}' "$tmp/out.$i" >> "$tmp/aval"
done

resumo() { # $1=arquivo $2=rotulo
  [ -s "$1" ] || { printf "%-24s (nao reportado)\n" "$2"; return; }
  # desvio = desvio padrao amostral (divide por n-1); 0 quando n=1.
  sort -n "$1" | awk -v r="$2" '{v[NR]=$1; s+=$1; s2+=$1*$1}
    END{m=(NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2; mu=s/NR;
        var=(NR>1)?(s2-NR*mu*mu)/(NR-1):0; if(var<0)var=0;
        printf "%-24s min=%.4f  mediana=%.4f  media=%.4f  max=%.4f  desvio=%.4f  (n=%d)\n", r, v[1], m, mu, v[NR], sqrt(var), NR}'
}
echo "== $BIN | repeticoes=$REP | args=${ARGS[*]:-<padrao>} =="
resumo "$tmp/treino" "Tempo de treino (s)"
resumo "$tmp/carga"  "Tempo de carga (s)"
resumo "$tmp/aval"   "Tempo de avaliacao (s)"
resumo "$tmp/wall"   "Tempo total/wall (s)"

# --- Corretude ---------------------------------------------------------
ok=1
if [ -f "$REF_SAIDA" ] && [ "$COMPARAR" -eq 1 ]; then
  for i in $(seq 1 "$REP"); do
    grep -v -E '^(Tempo|HASH|INFO)' "$tmp/out.$i" | diff -q - "$REF_SAIDA" > /dev/null || ok=0
  done
  [ $ok -eq 1 ] && echo "Saida textual : OK (identica a ref/saida_referencia.txt em todas as repeticoes)" \
                || echo "Saida textual : DIVERGE de ref/saida_referencia.txt"
else
  echo "Saida textual : (sem comparacao: ref ausente ou argumentos customizados)"
fi
if [ -f "$REF_HASH" ] && [ "$COMPARAR" -eq 1 ] && grep -q "^HASH_" "$tmp/out.1"; then
  hk=1
  for i in $(seq 1 "$REP"); do
    grep -E '^HASH_' "$tmp/out.$i" | diff -q - "$REF_HASH" > /dev/null || hk=0
  done
  [ $hk -eq 1 ] && echo "Hash bit a bit: OK (pesos e perdas identicos ao legado)" \
                || echo "Hash bit a bit: DIVERGE ($(grep -E '^HASH_' "$tmp/out.1" | tr '\n' ' '))"
fi

[ -f "$REF_HASH" ] && [ "$COMPARAR" -eq 1 ] && ! grep -q "^HASH_" "$tmp/out.1" && echo "Hash bit a bit: (binario sem instrumentacao de hash)"
exit 0
