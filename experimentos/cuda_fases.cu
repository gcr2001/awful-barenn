// Experimento: quantos ciclos de GPU cada FASE do passo de SGD consome?
//
// O Nsight Compute não consegue ler contadores de hardware no cluster Franky
// (ERR_NVGPUCTRPERM), então medimos por dentro do kernel: a thread 0 lê
// clock64() após cada __syncthreads() e acumula o intervalo de cada fase.
// O kernel é o persistente da cuda_v3 (pesos em shared, 1 bloco), rodando 1
// época. O custo do próprio clock64() (~30 ciclos/leitura) está incluído.
//
// Compilar: nvcc -O3 -arch=sm_86 --fmad=false -DTH=128 -Isrc \
//             experimentos/cuda_fases.cu -o bin/exp_fases
#include "../src/cuda_passo.cuh"

#ifndef TH
#define TH 128
#endif

using namespace std;
using namespace Barenn;

constexpr int NFASES = 10;
__device__ unsigned long long g_ciclos[NFASES];

template <int T>
__global__ void k_fases(const float* __restrict__ imagens, const int* __restrict__ rotulos,
                        int n, float lr, float* gWt1, float* gb1, float* gWt2, float* gb2,
                        float* gWt3, float* gb3) {
    __shared__ float sWt1[TAM_WT1], sb1[N1], sWt2[TAM_WT2], sb2[N2], sWt3[TAM_WT3], sb3[N3];
    __shared__ Compartilhado S;
    __shared__ float sx[2][N0];
    const int tid = threadIdx.x;
    for (int i = tid; i < TAM_WT1; i += T) sWt1[i] = gWt1[i];
    for (int i = tid; i < TAM_WT2; i += T) sWt2[i] = gWt2[i];
    for (int i = tid; i < TAM_WT3; i += T) sWt3[i] = gWt3[i];
    if (tid < N1) sb1[tid] = gb1[tid];
    if (tid < N2) sb2[tid] = gb2[tid];
    if (tid < N3) sb3[tid] = gb3[tid];
    if (tid < N0) sx[0][tid] = imagens[tid];
    int rot = rotulos[0];
    __syncthreads();

    unsigned long long ciclos[NFASES] = {0};
    double acc = 0;
    for (int k = 0; k < n; ++k) {
        const int cur = k & 1;
        const bool tem = (k + 1 < n);
        float pre = 0; int prox = 0;
        if (tem) { if (tid < N0) pre = imagens[(size_t)(k + 1) * N0 + tid]; prox = rotulos[k + 1]; }
        const float* x = sx[cur];
        long long t = clock64();
        auto marca = [&](int f) { if (tid == 0) { long long u = clock64(); ciclos[f] += u - t; t = u; } };

        if (tid < N1) { float sum = sb1[tid];
            #pragma unroll 8
            for (int j = 0; j < N0; ++j) sum += x[j] * sWt1[j * P1 + tid];
            S.z1[tid] = sum; S.h1[tid] = relu_(sum); }
        __syncthreads(); marca(0);                                   // F0: forward L1
        if (tid < N2) { float sum = sb2[tid];
            #pragma unroll 8
            for (int j = 0; j < N1; ++j) sum += S.h1[j] * sWt2[j * P2 + tid];
            S.z2[tid] = sum; S.h2[tid] = relu_(sum); }
        __syncthreads(); marca(1);                                   // F1: forward L2
        if (tid < N3) { float sum = sb3[tid];
            #pragma unroll
            for (int j = 0; j < N2; ++j) sum += S.h2[j] * sWt3[j * P3 + tid];
            S.s[tid] = sum; }
        __syncthreads(); marca(2);                                   // F2: forward L3
        if (tid == 0) { float total = 0;
            #pragma unroll
            for (int i = 0; i < N3; ++i) { const float e = S.s[i] - (i == rot ? 1.0f : 0.0f);
                total += 0.5f * e * e; S.g[i] = e; }
            acc += total; }
        __syncthreads(); marca(3);                                   // F3: perda
        if (tid < N2) { float sum = 0;
            #pragma unroll
            for (int j = 0; j < N3; ++j) sum += S.g[j] * sWt3[tid * P3 + j];
            S.e3[tid] = sum; }
        __syncthreads(); marca(4);                                   // F4: grad_in L3
        for (int idx = tid; idx < N2 * N3; idx += T) { const int j = idx / N3, kk = idx % N3;
            sWt3[j * P3 + kk] -= lr * (S.g[kk] * S.h2[j]); }
        if (tid < N3) sb3[tid] -= lr * S.g[tid];
        if (tid < N2) S.d2[tid] = S.e3[tid] * (S.z2[tid] > 0.0f ? 1.0f : 0.0f);
        __syncthreads(); marca(5);                                   // F5: update L3 + delta2
        if (tid < N1) { float sum = 0;
            #pragma unroll
            for (int j = 0; j < N2; ++j) sum += S.d2[j] * sWt2[tid * P2 + j];
            S.gi2[tid] = sum; }
        __syncthreads(); marca(6);                                   // F6: grad_in L2
        for (int idx = tid; idx < N1 * N2; idx += T) { const int j = idx / N2, kk = idx % N2;
            sWt2[j * P2 + kk] -= lr * (S.d2[kk] * S.h1[j]); }
        if (tid < N2) sb2[tid] -= lr * S.d2[tid];
        if (tid < N1) S.d1[tid] = S.gi2[tid] * (S.z1[tid] > 0.0f ? 1.0f : 0.0f);
        __syncthreads(); marca(7);                                   // F7: update L2 + delta1
        for (int idx = tid; idx < N0 * N1; idx += T) { const int j = idx / N1, kk = idx % N1;
            sWt1[j * P1 + kk] -= lr * (S.d1[kk] * x[j]); }
        if (tid < N1) sb1[tid] -= lr * S.d1[tid];
        __syncthreads(); marca(8);                                   // F8: update L1
        if (tem) { if (tid < N0) sx[cur ^ 1][tid] = pre; rot = prox; }
        __syncthreads(); marca(9);                                   // F9: commit do prefetch
    }
    if (tid == 0) for (int f = 0; f < NFASES; ++f) g_ciclos[f] = ciclos[f];
}

int main() {
    const Conjunto treino = para_conjunto(ler_dados("optdigits.tra"));
    const int n = static_cast<int>(treino.size());
    PesosHost pesos = PesosHost::iniciais();
    PesosGpu g = empacotar(pesos);
    Dev<float> d_img(treino.imagens.size()), d_wt1(TAM_WT1), d_b1(N1), d_wt2(TAM_WT2),
        d_b2(N2), d_wt3(TAM_WT3), d_b3(N3);
    Dev<int> d_rot(treino.rotulos.size());
    d_img.para_device(treino.imagens.data()); d_rot.para_device(treino.rotulos.data());
    d_wt1.para_device(g.wt1.data()); d_b1.para_device(pesos.b1.data());
    d_wt2.para_device(g.wt2.data()); d_b2.para_device(pesos.b2.data());
    d_wt3.para_device(g.wt3.data()); d_b3.para_device(pesos.b3.data());
    k_fases<TH><<<1, TH>>>(d_img.get(), d_rot.get(), n, taxa, d_wt1.get(), d_b1.get(),
                           d_wt2.get(), d_b2.get(), d_wt3.get(), d_b3.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long h[NFASES];
    CUDA_CHECK(cudaMemcpyFromSymbol(h, g_ciclos, sizeof h));
    const char* nome[NFASES] = {"forward camada 1 (64->32)", "forward camada 2 (32->16)",
        "forward camada 3 (16->10)", "perda + gradiente (1 thread)", "grad_in camada 3",
        "atualiza camada 3 + delta2", "grad_in camada 2", "atualiza camada 2 + delta1",
        "atualiza camada 1 (2048 pesos)", "commit do prefetch"};
    unsigned long long total = 0;
    for (int f = 0; f < NFASES; ++f) total += h[f];
    printf("TH=%d  ciclos medios por amostra (1 epoca, %d amostras):\n", TH, n);
    for (int f = 0; f < NFASES; ++f)
        printf("  F%d %-32s %7.0f ciclos  %5.1f%%\n", f, nome[f], (double)h[f] / n, 100.0 * h[f] / total);
    printf("  TOTAL %.0f ciclos/amostra  = %.3f us @1462 MHz\n", (double)total / n, total / (double)n / 1462.0);
    return 0;
}
