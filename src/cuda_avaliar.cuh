// Avaliação na GPU: um thread por imagem de teste, com a mesma ordem de soma
// do legado (resultado idêntico bit a bit). Usa o layout empacotado dos pesos.
#pragma once

#include "cuda_passo.cuh"

namespace Barenn {

__global__ void k_avaliar(const float* __restrict__ x, int nt,
                          const float* __restrict__ Wt1, const float* __restrict__ b1,
                          const float* __restrict__ Wt2, const float* __restrict__ b2,
                          const float* __restrict__ Wt3, const float* __restrict__ b3,
                          float* __restrict__ escores) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt) return;
    const float* xi = x + static_cast<size_t>(t) * N0;
    float h1[N1], h2[N2];
    for (int i = 0; i < N1; ++i) {
        float sum = b1[i];
        for (int j = 0; j < N0; ++j) sum += xi[j] * Wt1[j * P1 + i];
        h1[i] = relu_(sum);
    }
    for (int i = 0; i < N2; ++i) {
        float sum = b2[i];
        for (int j = 0; j < N1; ++j) sum += h1[j] * Wt2[j * P2 + i];
        h2[i] = relu_(sum);
    }
    for (int i = 0; i < N3; ++i) {
        float sum = b3[i];
        for (int j = 0; j < N2; ++j) sum += h2[j] * Wt3[j * P3 + i];
        escores[static_cast<size_t>(t) * N3 + i] = sum;
    }
}

}  // namespace Barenn
