#!/usr/bin/env python3
"""Gera build/cuda_v1_graph.cu: a cuda_v1 com o passo de uma amostra capturado
num CUDA Graph e relancado n vezes por epoca.

Hipotese testada: o gargalo da v1 e o custo de LANCAMENTO (CPU + driver), entao
um grafo (1 lancamento por amostra em vez de 10) deveria ajudar muito.
Previsao baseada no nsys: so remove a ociosidade entre kernels (~27%), pois o
piso de ~1,2 us por kernel continua valendo dentro do grafo (<= ~1,3x).

Uso (raiz do repo): python3 experimentos/gerar_cuda_v1_graph.py
Compilar:  nvcc -O3 -arch=sm_86 --fmad=false -Xcompiler -ffp-contract=off \
                -Isrc build/cuda_v1_graph.cu -o bin/exp_cuda_v1_graph
"""
import os
import re

src = open("src/cuda_v1.cu").read()

# 1) todos os lancamentos de kernel passam a usar a stream `st`
src = re.sub(r"<<<([^<>]*?)>>>", r"<<<\1, 0, st>>>", src)

# 2) captura do passo (10 lancamentos) num grafo e relancamento n vezes
a = src.index("            for (int k = 0; k < n; ++k) {")
b = src.index("            CUDA_CHECK(cudaGetLastError());\n            double perda_total = 0;")
corpo = src[a:b]
ini = corpo.index("{") + 1
fim = corpo.rindex("}")
lancamentos = corpo[ini:fim]

captura = (
    "        cudaStream_t st;\n"
    "        CUDA_CHECK(cudaStreamCreate(&st));\n"
    "        cudaGraph_t grafo;\n"
    "        cudaGraphExec_t exec;\n"
    "        CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));\n"
    + lancamentos +
    "        CUDA_CHECK(cudaStreamEndCapture(st, &grafo));\n"
    "        CUDA_CHECK(cudaGraphInstantiate(&exec, grafo, 0));\n"
)
novo_laco = (
    "            for (int k = 0; k < n; ++k) CUDA_CHECK(cudaGraphLaunch(exec, st));\n"
    "            CUDA_CHECK(cudaStreamSynchronize(st));\n"
)
src = src[:a] + novo_laco + src[b:]
# a captura precisa vir antes do laco de epocas (e antes do cronometro de treino)
marca = "        Cronometro cron_treino;"
src = src.replace(marca, captura + marca, 1)
# d_cursor/d_perda.zerar() usam memset na stream padrao; ok com stream bloqueante.

os.makedirs("build", exist_ok=True)
open("build/cuda_v1_graph.cu", "w").write(
    "// GERADO por experimentos/gerar_cuda_v1_graph.py - nao editar.\n" + src)
print("Gerado build/cuda_v1_graph.cu")
