#!/usr/bin/env bash
# Gera uma COPIA instrumentada do codigo legado (barenn.cpp) que, quando a
# variavel de ambiente BARENN_HASH=1 esta definida, imprime dois hashes:
#   HASH_PESOS  - FNV-1a de todos os pesos e vieses finais (ordem canonica
#                 camada 1,2,3; peso [saida][entrada]; depois vies)
#   HASH_PERDAS - FNV-1a das perdas medias por epoca (double, bit a bit)
# O barenn.cpp original NAO e alterado. As versoes otimizadas implementam o
# mesmo hash nativamente, o que permite provar que o resultado e identico
# bit a bit ao do legado (e nao apenas "parecido" nas 4 casas impressas).
#
# Uso: scripts/instrumentar_baseline.sh [saida.cpp]
set -euo pipefail
cd "$(dirname "$0")/.."
SAIDA="${1:-build/barenn_ref_hash.cpp}"
mkdir -p "$(dirname "$SAIDA")"

METODO=$(mktemp)
MAIN=$(mktemp)
INCS=$(mktemp)
printf '#include <cstdio>\n#include <cstdlib>\n' > "$INCS"
cat > "$METODO" <<'EOF'
    static void HashMix(unsigned long long& h, float v) {
        unsigned int b;
        __builtin_memcpy(&b, &v, sizeof b);
        for (int k = 0; k < 4; ++k) { h ^= (b >> (8 * k)) & 0xFFu; h *= 1099511628211ULL; }
    }
    void HashInto(unsigned long long& h) const {
        for (const auto& row : weights) for (float w : row) HashMix(h, w);
        for (float b : bias) HashMix(h, b);
    }
EOF
cat > "$MAIN" <<'EOF'
        if (getenv("BARENN_HASH")) {
            unsigned long long hp = 1469598103934665603ULL;
            l1.HashInto(hp); l2.HashInto(hp); l3.HashInto(hp);
            unsigned long long hl = 1469598103934665603ULL;
            for (double p : perdas) {
                unsigned long long b; __builtin_memcpy(&b, &p, sizeof b);
                for (int k = 0; k < 8; ++k) { hl ^= (b >> (8 * k)) & 0xFFull; hl *= 1099511628211ULL; }
            }
            printf("HASH_PESOS %016llx\nHASH_PERDAS %016llx\n", hp, hl);
        }
EOF

# 1) injeta o metodo logo apos o "public:" da DenseLayer
# 2) injeta o calculo do hash logo apos a linha que imprime o tempo de treino
# (sintaxe portavel entre sed GNU e BSD)
sed -e "/^public:/r $METODO" \
    -e "/Tempo de treino/r $MAIN" \
    -e "/^#include <vector>/r $INCS" \
    barenn.cpp > "$SAIDA"
rm -f "$METODO" "$MAIN" "$INCS"
echo "Gerado: $SAIDA"
