// Um passo completo de SGD (forward + perda + backward + atualização) executado
// por UM bloco de threads, com as ativações em memória compartilhada.
//
// Por que um bloco só? O SGD é sequencial entre amostras e, dentro de uma
// amostra, as camadas formam uma cadeia de dependências. O paralelismo útil é
// pequeno (<= 64 neurônios independentes por etapa), então o que importa é a
// LATÊNCIA do passo, não a vazão: um bloco, sincronizando com __syncthreads()
// (~20-30 ciclos) em vez de lançar kernels (~1,2 us cada, medido com nsys).
//
// Exatidão numérica: cada neurônio é somado por UMA thread, em ordem crescente
// de entrada, como no legado (nenhuma redução em árvore). Compilar com
// --fmad=false. O resultado é idêntico bit a bit ao da CPU.
//
// Layout dos pesos na GPU ([entrada][saida], saída contígua):
//   Wt1[j*32 + i]   (camada 1: 64 x 32)
//   Wt2[j*17 + i]   (camada 2: 32 x 16, pitch 17)
//   Wt3[j*11 + i]   (camada 3: 16 x 10, pitch 11)
// No forward, threads consecutivas (i) leem endereços consecutivos. No
// grad_in a thread é a ENTRADA (j) e percorre as saídas: com pitch ímpar
// (17, 11) os acessos j*pitch caem em bancos de memória distintos, sem
// conflito. Assim uma única cópia dos pesos serve aos dois sentidos.
#pragma once

#include "cuda_util.cuh"

namespace Barenn {

constexpr int P1 = N1;   // 32
constexpr int P2 = 17;   // N2 + 1
constexpr int P3 = 11;   // N3 + 1

constexpr int TAM_WT1 = N0 * P1;  // 2048
constexpr int TAM_WT2 = N1 * P2;  // 544
constexpr int TAM_WT3 = N2 * P3;  // 176

// Pesos no layout da GPU (host).
struct PesosGpu {
    vector<float> wt1, wt2, wt3;
};

// canônico [saida][entrada]  ->  [entrada][saida] com pitch
inline PesosGpu empacotar(const PesosHost& p) {
    PesosGpu g;
    g.wt1.assign(TAM_WT1, 0.0f);
    g.wt2.assign(TAM_WT2, 0.0f);
    g.wt3.assign(TAM_WT3, 0.0f);
    for (int i = 0; i < N1; ++i) for (int j = 0; j < N0; ++j) g.wt1[j * P1 + i] = p.w1[i * N0 + j];
    for (int i = 0; i < N2; ++i) for (int j = 0; j < N1; ++j) g.wt2[j * P2 + i] = p.w2[i * N1 + j];
    for (int i = 0; i < N3; ++i) for (int j = 0; j < N2; ++j) g.wt3[j * P3 + i] = p.w3[i * N2 + j];
    return g;
}

// inverso de empacotar (os vieses não mudam de layout)
inline void desempacotar(const PesosGpu& g, PesosHost& p) {
    for (int i = 0; i < N1; ++i) for (int j = 0; j < N0; ++j) p.w1[i * N0 + j] = g.wt1[j * P1 + i];
    for (int i = 0; i < N2; ++i) for (int j = 0; j < N1; ++j) p.w2[i * N1 + j] = g.wt2[j * P2 + i];
    for (int i = 0; i < N3; ++i) for (int j = 0; j < N2; ++j) p.w3[i * N2 + j] = g.wt3[j * P3 + i];
}

// Ativações e gradientes do passo atual (ficam em memória compartilhada).
struct Compartilhado {
    float z1[N1], h1[N1];
    float z2[N2], h2[N2];
    float s[N3], g[N3];
    float e3[N2];   // grad_in da camada 3
    float d2[N2];   // delta da camada 2
    float gi2[N1];  // grad_in da camada 2
    float d1[N1];   // delta da camada 1
};

__device__ __forceinline__ float relu_(float s) { return (0.0f < s) ? s : 0.0f; }

// Todas as threads do bloco chamam esta função (há __syncthreads dentro).
// `x`: 64 entradas (global ou compartilhada). `perda_acum`: só a thread 0 o
// altera (soma em double, na ordem das amostras, como o legado).
template <int NT>
__device__ __forceinline__ void passo_amostra(
    const float* __restrict__ x, int rotulo, float lr,
    float* Wt1, float* b1, float* Wt2, float* b2, float* Wt3, float* b3,
    Compartilhado& S, double& perda_acum) {
    const int tid = threadIdx.x;

    // ---------------- forward ----------------
    if (tid < N1) {
        float sum = b1[tid];
        #pragma unroll 8
        for (int j = 0; j < N0; ++j) sum += x[j] * Wt1[j * P1 + tid];
        S.z1[tid] = sum;
        S.h1[tid] = relu_(sum);
    }
    __syncthreads();
    if (tid < N2) {
        float sum = b2[tid];
        #pragma unroll 8
        for (int j = 0; j < N1; ++j) sum += S.h1[j] * Wt2[j * P2 + tid];
        S.z2[tid] = sum;
        S.h2[tid] = relu_(sum);
    }
    __syncthreads();
    if (tid < N3) {
        float sum = b3[tid];
        #pragma unroll
        for (int j = 0; j < N2; ++j) sum += S.h2[j] * Wt3[j * P3 + tid];
        S.s[tid] = sum;  // camada final: escores lineares
    }
    __syncthreads();

    // ------- perda e gradiente nos escores (mesma ordem de soma do legado) -------
    if (tid == 0) {
        float total = 0;
        #pragma unroll
        for (int i = 0; i < N3; ++i) {
            const float erro = S.s[i] - (i == rotulo ? 1.0f : 0.0f);
            total += 0.5f * erro * erro;
            S.g[i] = erro;
        }
        perda_acum += total;
    }
    __syncthreads();

    // ------- camada 3: grad_in com os pesos ANTES da atualização -------
    if (tid < N2) {
        float sum = 0;
        #pragma unroll
        for (int j = 0; j < N3; ++j) sum += S.g[j] * Wt3[tid * P3 + j];
        S.e3[tid] = sum;
    }
    __syncthreads();

    // ------- atualização da camada 3  ||  delta da camada 2 -------
    for (int idx = tid; idx < N2 * N3; idx += NT) {
        const int j = idx / N3, k = idx % N3;
        Wt3[j * P3 + k] -= lr * (S.g[k] * S.h2[j]);
    }
    if (tid < N3) b3[tid] -= lr * S.g[tid];
    if (tid < N2) S.d2[tid] = S.e3[tid] * (S.z2[tid] > 0.0f ? 1.0f : 0.0f);
    __syncthreads();

    // ------- camada 2: grad_in com os pesos ANTES da atualização -------
    if (tid < N1) {
        float sum = 0;
        #pragma unroll
        for (int j = 0; j < N2; ++j) sum += S.d2[j] * Wt2[tid * P2 + j];
        S.gi2[tid] = sum;
    }
    __syncthreads();

    // ------- atualização da camada 2  ||  delta da camada 1 -------
    for (int idx = tid; idx < N1 * N2; idx += NT) {
        const int j = idx / N2, k = idx % N2;
        Wt2[j * P2 + k] -= lr * (S.d2[k] * S.h1[j]);
    }
    if (tid < N2) b2[tid] -= lr * S.d2[tid];
    if (tid < N1) S.d1[tid] = S.gi2[tid] * (S.z1[tid] > 0.0f ? 1.0f : 0.0f);
    __syncthreads();

    // ------- atualização da camada 1 (o grad_in da entrada não é usado) -------
    for (int idx = tid; idx < N0 * N1; idx += NT) {
        const int j = idx / N1, k = idx % N1;
        Wt1[j * P1 + k] -= lr * (S.d1[k] * x[j]);
    }
    if (tid < N1) b1[tid] -= lr * S.d1[tid];
    __syncthreads();  // o forward da próxima amostra lê Wt1
}

}  // namespace Barenn
