// Barenn - CPU v2: layout SIMD com dimensões em tempo de compilação.
//
// Resultado idêntico, bit a bit, ao do legado. A ideia central é vetorizar
// ENTRE NEURÔNIOS e nunca reassociar a soma de um mesmo neurônio:
//   forward : out[i] = bias[i] + sum_j x[j]*w[i][j]   (j crescente)
// Guardando os pesos como Wt[j][i] (i contíguo), o laço em j faz
//   acc[0..N) = acc[0..N) + x[j] * Wt[j][0..N)
// ou seja, cada lane i soma suas parcelas na MESMA ordem do legado, mas 8
// neurônios avançam por instrução. O compilador não pode fazer isso sozinho
// (exigiria reassociar a soma, proibido sem -ffast-math).
//
// Para o grad_in precisamos da orientação oposta W[j][i] (i contíguo), então
// as camadas 2 e 3 mantêm as duas orientações (as atualizações são elemento a
// elemento e custam pouco). A camada 1 só precisa de Wt: seu grad_in nunca é
// usado.
//
// Compilar SEM FMA para preservar o arredondamento: -ffp-contract=off.
#pragma once
#include "common.hpp"

namespace Barenn {

constexpr int N0 = entradas;  // 64
constexpr int N1 = 32;
constexpr int N2 = 16;
constexpr int N3 = classes;   // 10
constexpr int P3 = 16;        // N3 completado até múltiplo de 8 (lanes extras = 0)

typedef float v8 __attribute__((vector_size(32)));
typedef int v8i __attribute__((vector_size(32)));

static inline v8 ld(const float* p) { v8 v; memcpy(&v, p, sizeof v); return v; }
static inline void st(float* p, v8 v) { memcpy(p, &v, sizeof v); }
static inline v8 bc(float x) { return v8{x, x, x, x, x, x, x, x}; }

// ReLU como no legado: max(0.0f, z) == (0 < z) ? z : +0.
static inline v8 relu(v8 z) {
    const v8i m = z > bc(0.0f);
    return (v8)(m & (v8i)z);
}
// grad * (z > 0 ? 1 : 0), mantendo a multiplicação do legado.
static inline v8 dmask(v8 grad, v8 z) {
    const v8i m = z > bc(0.0f);
    return grad * (v8)(m & (v8i)bc(1.0f));
}

struct Ativ {
    alignas(32) float z1[N1];
    alignas(32) float h1[N1];
    alignas(32) float z2[N2];
    alignas(32) float h2[N2];
    alignas(32) float s[P3];
};

struct Rede {
    alignas(32) float Wt1[N0 * N1];  // [entrada][saida]
    alignas(32) float b1[N1];
    alignas(32) float Wt2[N1 * N2];  // [entrada][saida]  (forward e atualização)
    alignas(32) float W2[N2 * N1];   // [saida][entrada]  (grad_in)
    alignas(32) float b2[N2];
    alignas(32) float Wt3[N2 * P3];  // [entrada][saida acolchoada]
    alignas(32) float W3[N3 * N2];   // [saida][entrada]
    alignas(32) float b3[P3];

    Rede() {
        memset(this, 0, sizeof *this);
        auto w1 = pesos_iniciais(N0, N1);
        auto w2 = pesos_iniciais(N1, N2);
        auto w3 = pesos_iniciais(N2, N3);
        for (int i = 0; i < N1; ++i)
            for (int j = 0; j < N0; ++j) Wt1[j * N1 + i] = w1[i * N0 + j];
        for (int i = 0; i < N2; ++i)
            for (int j = 0; j < N1; ++j) {
                Wt2[j * N2 + i] = w2[i * N1 + j];
                W2[i * N1 + j] = w2[i * N1 + j];
            }
        for (int i = 0; i < N3; ++i)
            for (int j = 0; j < N2; ++j) {
                Wt3[j * P3 + i] = w3[i * N2 + j];
                W3[i * N2 + j] = w3[i * N2 + j];
            }
    }

    // Forward sem estado: pode ser chamado em paralelo (rede só é lida).
    void forward(const float* __restrict x, Ativ& a) const {
        v8 c0 = ld(b1), c1 = ld(b1 + 8), c2 = ld(b1 + 16), c3 = ld(b1 + 24);
        for (int j = 0; j < N0; ++j) {
            const v8 xj = bc(x[j]);
            const float* w = Wt1 + j * N1;
            c0 = c0 + xj * ld(w);
            c1 = c1 + xj * ld(w + 8);
            c2 = c2 + xj * ld(w + 16);
            c3 = c3 + xj * ld(w + 24);
        }
        st(a.z1, c0); st(a.z1 + 8, c1); st(a.z1 + 16, c2); st(a.z1 + 24, c3);
        st(a.h1, relu(c0)); st(a.h1 + 8, relu(c1));
        st(a.h1 + 16, relu(c2)); st(a.h1 + 24, relu(c3));

        v8 d0 = ld(b2), d1 = ld(b2 + 8);
        for (int j = 0; j < N1; ++j) {
            const v8 hj = bc(a.h1[j]);
            const float* w = Wt2 + j * N2;
            d0 = d0 + hj * ld(w);
            d1 = d1 + hj * ld(w + 8);
        }
        st(a.z2, d0); st(a.z2 + 8, d1);
        st(a.h2, relu(d0)); st(a.h2 + 8, relu(d1));

        v8 s0 = ld(b3), s1 = ld(b3 + 8);  // camada final: escores lineares
        for (int j = 0; j < N2; ++j) {
            const v8 hj = bc(a.h2[j]);
            const float* w = Wt3 + j * P3;
            s0 = s0 + hj * ld(w);
            s1 = s1 + hj * ld(w + 8);
        }
        st(a.s, s0); st(a.s + 8, s1);
    }

    // Um passo de SGD (forward, perda, backward, atualização). Devolve a perda.
    float passo(const float* __restrict x, int rotulo, float lr) {
        Ativ a;
        forward(x, a);

        // Perda e gradiente nos escores (mesma ordem de soma do legado).
        alignas(32) float g[P3] = {0};
        float perda_amostra = 0;
        for (int i = 0; i < N3; ++i) {
            const float erro = a.s[i] - (i == rotulo ? 1.0f : 0.0f);
            perda_amostra += 0.5f * erro * erro;
            g[i] = erro;
        }
        const v8 lrv = bc(lr);

        // ---- camada 3 (final): delta = gradiente dos escores ----------------
        const v8 g0 = ld(g), g1 = ld(g + 8);
        v8 e0 = bc(0.0f), e1 = bc(0.0f);  // grad_in da camada 3 (pesos antigos)
        for (int j = 0; j < N3; ++j) {
            const v8 dj = bc(g[j]);
            e0 = e0 + dj * ld(W3 + j * N2);
            e1 = e1 + dj * ld(W3 + j * N2 + 8);
        }
        {
            const v8 ha = ld(a.h2), hb = ld(a.h2 + 8);
            for (int k = 0; k < N3; ++k) {
                const v8 dk = bc(g[k]);
                float* w = W3 + k * N2;
                st(w, ld(w) - lrv * (dk * ha));
                st(w + 8, ld(w + 8) - lrv * (dk * hb));
            }
            for (int j = 0; j < N2; ++j) {
                const v8 xj = bc(a.h2[j]);
                float* w = Wt3 + j * P3;
                st(w, ld(w) - lrv * (g0 * xj));
                st(w + 8, ld(w + 8) - lrv * (g1 * xj));
            }
            st(b3, ld(b3) - lrv * g0);
            st(b3 + 8, ld(b3 + 8) - lrv * g1);
        }

        // ---- camada 2 ----------------------------------------------------------
        const v8 q0 = dmask(e0, ld(a.z2)), q1 = dmask(e1, ld(a.z2 + 8));
        alignas(32) float d2[N2];
        st(d2, q0); st(d2 + 8, q1);
        v8 f0 = bc(0.0f), f1 = bc(0.0f), f2 = bc(0.0f), f3 = bc(0.0f);
        for (int j = 0; j < N2; ++j) {  // grad_in da camada 2 (pesos antigos)
            const v8 dj = bc(d2[j]);
            const float* w = W2 + j * N1;
            f0 = f0 + dj * ld(w);
            f1 = f1 + dj * ld(w + 8);
            f2 = f2 + dj * ld(w + 16);
            f3 = f3 + dj * ld(w + 24);
        }
        {
            const v8 ha = ld(a.h1), hb = ld(a.h1 + 8), hc = ld(a.h1 + 16), hd = ld(a.h1 + 24);
            for (int k = 0; k < N2; ++k) {
                const v8 dk = bc(d2[k]);
                float* w = W2 + k * N1;
                st(w, ld(w) - lrv * (dk * ha));
                st(w + 8, ld(w + 8) - lrv * (dk * hb));
                st(w + 16, ld(w + 16) - lrv * (dk * hc));
                st(w + 24, ld(w + 24) - lrv * (dk * hd));
            }
            for (int j = 0; j < N1; ++j) {
                const v8 xj = bc(a.h1[j]);
                float* w = Wt2 + j * N2;
                st(w, ld(w) - lrv * (q0 * xj));
                st(w + 8, ld(w + 8) - lrv * (q1 * xj));
            }
            st(b2, ld(b2) - lrv * q0);
            st(b2 + 8, ld(b2 + 8) - lrv * q1);
        }

        // ---- camada 1 (grad_in desnecessário) -------------------------------------
        const v8 r0 = dmask(f0, ld(a.z1)), r1 = dmask(f1, ld(a.z1 + 8));
        const v8 r2 = dmask(f2, ld(a.z1 + 16)), r3 = dmask(f3, ld(a.z1 + 24));
        for (int j = 0; j < N0; ++j) {
            const v8 xj = bc(x[j]);
            float* w = Wt1 + j * N1;
            st(w, ld(w) - lrv * (r0 * xj));
            st(w + 8, ld(w + 8) - lrv * (r1 * xj));
            st(w + 16, ld(w + 16) - lrv * (r2 * xj));
            st(w + 24, ld(w + 24) - lrv * (r3 * xj));
        }
        st(b1, ld(b1) - lrv * r0);
        st(b1 + 8, ld(b1 + 8) - lrv * r1);
        st(b1 + 16, ld(b1 + 16) - lrv * r2);
        st(b1 + 24, ld(b1 + 24) - lrv * r3);

        return perda_amostra;
    }

    // Ordem canônica do legado: por camada, pesos [saida][entrada] e depois viés.
    void hash_into(Fnv& f) const {
        for (int i = 0; i < N1; ++i) for (int j = 0; j < N0; ++j) f.mix(Wt1[j * N1 + i]);
        for (int i = 0; i < N1; ++i) f.mix(b1[i]);
        for (int i = 0; i < N2; ++i) for (int j = 0; j < N1; ++j) f.mix(W2[i * N1 + j]);
        for (int i = 0; i < N2; ++i) f.mix(b2[i]);
        for (int i = 0; i < N3; ++i) for (int j = 0; j < N2; ++j) f.mix(W3[i * N2 + j]);
        for (int i = 0; i < N3; ++i) f.mix(b3[i]);
    }
};

}  // namespace Barenn
