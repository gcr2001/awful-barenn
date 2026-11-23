#!/usr/bin/env python3
"""Gera build/cuda_v4_fases.cu: a cuda_v4 instrumentada com clock64().

Injeta, apos cada __syncthreads() do laco de amostras, a leitura de clock64()
pela thread 0, acumulando os ciclos de cada fase. Nao altera a aritmetica.
Uso (na raiz do repo):
    python3 experimentos/gerar_fases_v4.py
    nvcc -O3 -arch=sm_86 --fmad=false -Xcompiler -ffp-contract=off -DTH=128 \
         -Isrc build/cuda_v4_fases.cu -o bin/exp_fases_v4
"""
import re

src = open("src/cuda_v4.cu").read()
ini = src.index("struct alignas(16) Estado")
fim = src.index("int main(")
kernel = src[ini:fim]

# Delimita o laco de amostras do kernel
a = kernel.index("for (int k = 0; k < n; ++k) {")
b = kernel.index("if (tid == TID_PERDA) perdas_out[epoca]")
laco = kernel[a:b]
cont = [0]
def marca(m):
    i = cont[0]; cont[0] += 1
    return "__syncthreads(); MARCA(%d);" % i
laco2 = re.sub(r"__syncthreads\(\);", marca, laco)
nbar = cont[0]
kernel = kernel[:a] + laco2 + kernel[b:]

# declaracoes antes do laco de amostras e despejo ao fim da epoca
kernel = kernel.replace("        for (int k = 0; k < n; ++k) {",
    "        unsigned long long ciclos[%d] = {0}; long long tmark = clock64();\n"
    "        for (int k = 0; k < n; ++k) {" % nbar, 1)
kernel = kernel.replace("        if (tid == TID_PERDA) perdas_out[epoca] = acc / n;",
    "        if (tid == TID_PERDA) perdas_out[epoca] = acc / n;\n"
    "        if (tid == 0) for (int f = 0; f < %d; ++f) g_ciclos[f] += ciclos[f];" % nbar, 1)

cab = '''// GERADO por experimentos/gerar_fases_v4.py - nao editar.
#include "cuda_avaliar.cuh"
#ifndef TH
#define TH 128
#endif
using namespace std;
using namespace Barenn;
__device__ unsigned long long g_ciclos[16];
#define MARCA(i) do { if (tid == 0) { long long u_ = clock64(); ciclos[i] += u_ - tmark; tmark = u_; } } while (0)
'''
corpo_main = '''
int main() {
    const Conjunto treino = para_conjunto(ler_dados("optdigits.tra"));
    const int n = static_cast<int>(treino.size());
    PesosHost pesos = PesosHost::iniciais();
    PesosGpu g = empacotar(pesos);
    Dev<float> d_img(treino.imagens.size()), d_wt1(TAM_WT1), d_b1(N1), d_wt2(TAM_WT2),
        d_b2(N2), d_wt3(TAM_WT3), d_b3(N3);
    Dev<int> d_rot(treino.rotulos.size());
    Dev<double> d_perdas(1);
    d_img.para_device(treino.imagens.data()); d_rot.para_device(treino.rotulos.data());
    d_wt1.para_device(g.wt1.data()); d_b1.para_device(pesos.b1.data());
    d_wt2.para_device(g.wt2.data()); d_b2.para_device(pesos.b2.data());
    d_wt3.para_device(g.wt3.data()); d_b3.para_device(pesos.b3.data());
    k_treino4<TH><<<1, TH>>>(d_img.get(), d_rot.get(), n, 1, taxa, d_wt1.get(), d_b1.get(),
        d_wt2.get(), d_b2.get(), d_wt3.get(), d_b3.get(), d_perdas.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long h[16];
    CUDA_CHECK(cudaMemcpyFromSymbol(h, g_ciclos, sizeof h));
    const char* nome[] = {"forward L1 (+barreira)", "forward L2", "forward L3",
        "A: grad_in L3 || perda", "B: atualiza L3 || grad_in L2", "C: atualiza L2 || delta1",
        "D: atualiza L1 + prefetch"};
    unsigned long long total = 0;
    for (int f = 0; f < %d; ++f) total += h[f];
    printf("TH=%%d  ciclos medios por amostra (1 epoca, %%d amostras):\\n", TH, n);
    for (int f = 0; f < %d; ++f)
        printf("  F%%d %%-32s %%7.0f ciclos  %%5.1f%%%%\\n", f, nome[f], (double)h[f] / n, 100.0 * h[f] / total);
    printf("  TOTAL %%.0f ciclos/amostra = %%.3f us @1462 MHz\\n", (double)total / n, total / (double)n / 1462.0);
    return 0;
}
''' % (nbar, nbar)

import os
os.makedirs("build", exist_ok=True)
open("build/cuda_v4_fases.cu", "w").write(cab + kernel + corpo_main)
print("Gerado build/cuda_v4_fases.cu com %d marcas" % nbar)
