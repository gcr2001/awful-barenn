// Experimento NEGATIVO: paralelizar POR DENTRO de cada amostra (OpenMP).
//
// Hipótese: a camada 1 (64x32, a mais pesada) pode ser dividida em 4 blocos de
// 8 neurônios, um por thread, tanto no forward quanto na atualização dos
// pesos. O resultado continua idêntico bit a bit (cada bloco é independente),
// mas são 2 regiões paralelas por amostra, ~390 mil por época-conjunto.
//
// Resultado (ver Issue de multicore): MUITO mais lento que a v2 serial, porque
// o fork/join custa mais que o trabalho distribuído. Este arquivo existe só
// para documentar e reproduzir o experimento; NÃO faz parte das versões
// entregues.
//
// Compilar: g++ -O3 -march=x86-64-v3 -ffp-contract=off -fopenmp \
//             -Isrc experimentos/cpu_omp_amostra.cpp -o bin/exp_omp_amostra
#include <omp.h>

#include "../src/rede_simd.hpp"

using namespace std;
using namespace Barenn;

// Mesmo passo da Rede::passo, mas com a camada 1 dividida entre `nt` threads.
static float passo_omp(Rede& r, const float* x, int rotulo, float lr, int nt) {
    Ativ a;
    // --- forward camada 1 em paralelo: 4 blocos de 8 neurônios ---
    #pragma omp parallel for num_threads(nt) schedule(static)
    for (int blk = 0; blk < 4; ++blk) {
        v8 c = ld(r.b1 + 8 * blk);
        for (int j = 0; j < N0; ++j) c = c + bc(x[j]) * ld(r.Wt1 + j * N1 + 8 * blk);
        st(a.z1 + 8 * blk, c);
        st(a.h1 + 8 * blk, relu(c));
    }
    // --- camadas 2 e 3: iguais à versão serial ---
    v8 d0 = ld(r.b2), d1 = ld(r.b2 + 8);
    for (int j = 0; j < N1; ++j) {
        const v8 hj = bc(a.h1[j]);
        d0 = d0 + hj * ld(r.Wt2 + j * N2);
        d1 = d1 + hj * ld(r.Wt2 + j * N2 + 8);
    }
    st(a.z2, d0); st(a.z2 + 8, d1);
    st(a.h2, relu(d0)); st(a.h2 + 8, relu(d1));
    v8 s0 = ld(r.b3), s1 = ld(r.b3 + 8);
    for (int j = 0; j < N2; ++j) {
        const v8 hj = bc(a.h2[j]);
        s0 = s0 + hj * ld(r.Wt3 + j * P3);
        s1 = s1 + hj * ld(r.Wt3 + j * P3 + 8);
    }
    st(a.s, s0); st(a.s + 8, s1);

    alignas(32) float g[P3] = {0};
    float perda_amostra = 0;
    for (int i = 0; i < N3; ++i) {
        const float erro = a.s[i] - (i == rotulo ? 1.0f : 0.0f);
        perda_amostra += 0.5f * erro * erro;
        g[i] = erro;
    }
    const v8 lrv = bc(lr);
    const v8 g0 = ld(g), g1 = ld(g + 8);
    v8 e0 = bc(0.0f), e1 = bc(0.0f);
    for (int j = 0; j < N3; ++j) {
        const v8 dj = bc(g[j]);
        e0 = e0 + dj * ld(r.W3 + j * N2);
        e1 = e1 + dj * ld(r.W3 + j * N2 + 8);
    }
    {
        const v8 ha = ld(a.h2), hb = ld(a.h2 + 8);
        for (int k = 0; k < N3; ++k) {
            const v8 dk = bc(g[k]);
            float* w = r.W3 + k * N2;
            st(w, ld(w) - lrv * (dk * ha));
            st(w + 8, ld(w + 8) - lrv * (dk * hb));
        }
        for (int j = 0; j < N2; ++j) {
            const v8 xj = bc(a.h2[j]);
            float* w = r.Wt3 + j * P3;
            st(w, ld(w) - lrv * (g0 * xj));
            st(w + 8, ld(w + 8) - lrv * (g1 * xj));
        }
        st(r.b3, ld(r.b3) - lrv * g0);
        st(r.b3 + 8, ld(r.b3 + 8) - lrv * g1);
    }
    const v8 q0 = dmask(e0, ld(a.z2)), q1 = dmask(e1, ld(a.z2 + 8));
    alignas(32) float d2[N2];
    st(d2, q0); st(d2 + 8, q1);
    v8 f0 = bc(0.0f), f1 = bc(0.0f), f2 = bc(0.0f), f3 = bc(0.0f);
    for (int j = 0; j < N2; ++j) {
        const v8 dj = bc(d2[j]);
        const float* w = r.W2 + j * N1;
        f0 = f0 + dj * ld(w); f1 = f1 + dj * ld(w + 8);
        f2 = f2 + dj * ld(w + 16); f3 = f3 + dj * ld(w + 24);
    }
    {
        const v8 ha = ld(a.h1), hb = ld(a.h1 + 8), hc = ld(a.h1 + 16), hd = ld(a.h1 + 24);
        for (int k = 0; k < N2; ++k) {
            const v8 dk = bc(d2[k]);
            float* w = r.W2 + k * N1;
            st(w, ld(w) - lrv * (dk * ha)); st(w + 8, ld(w + 8) - lrv * (dk * hb));
            st(w + 16, ld(w + 16) - lrv * (dk * hc)); st(w + 24, ld(w + 24) - lrv * (dk * hd));
        }
        for (int j = 0; j < N1; ++j) {
            const v8 xj = bc(a.h1[j]);
            float* w = r.Wt2 + j * N2;
            st(w, ld(w) - lrv * (q0 * xj)); st(w + 8, ld(w + 8) - lrv * (q1 * xj));
        }
        st(r.b2, ld(r.b2) - lrv * q0); st(r.b2 + 8, ld(r.b2 + 8) - lrv * q1);
    }
    const v8 rr[4] = {dmask(f0, ld(a.z1)), dmask(f1, ld(a.z1 + 8)),
                      dmask(f2, ld(a.z1 + 16)), dmask(f3, ld(a.z1 + 24))};
    // --- atualização camada 1 em paralelo: cada thread cuida de 8 neurônios ---
    #pragma omp parallel for num_threads(nt) schedule(static)
    for (int blk = 0; blk < 4; ++blk) {
        for (int j = 0; j < N0; ++j) {
            const v8 xj = bc(x[j]);
            float* w = r.Wt1 + j * N1 + 8 * blk;
            st(w, ld(w) - lrv * (rr[blk] * xj));
        }
        st(r.b1 + 8 * blk, ld(r.b1 + 8 * blk) - lrv * rr[blk]);
    }
    return perda_amostra;
}

int main(int argc, char** argv) {
    const int nt = argc > 1 ? atoi(argv[1]) : 2;
    const Conjunto dados = para_conjunto(ler_dados("optdigits.tra"));
    auto rede = make_unique<Rede>();
    const size_t n = dados.size();
    array<double, epocas> perdas{};
    Cronometro cron;
    for (int ep = 0; ep < epocas; ++ep) {
        double total = 0;
        for (size_t k = 0; k < n; ++k)
            total += passo_omp(*rede, &dados.imagens[k * entradas], dados.rotulos[k], taxa, nt);
        perdas[ep] = total / n;
    }
    const double t = cron.segundos();
    Fnv h;
    rede->hash_into(h);
    printf("threads=%d  Tempo de treino %.4f s  (HASH_PESOS %016llx)\n", nt, t,
           static_cast<unsigned long long>(h.h));
    return 0;
}
