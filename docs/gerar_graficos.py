#!/usr/bin/env python3
"""Gera os graficos de docs/img/ a partir dos resultados medidos no cluster Franky.

Os numeros vem de docs/evidencias/ (mediana de 9 execucoes em no exclusivo, todas
as versoes de uma tabela medidas NA MESMA ALOCACAO). Uso: python3 docs/gerar_graficos.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("docs/img", exist_ok=True)
COR = {"legado": "#7f7f7f", "cpu": "#1f77b4", "cuda": "#2ca02c"}

# ----------------------------------------------------------------------------
# 1) Tempo de treino por versao (no de GPU: host i9-14900 + RTX A1000)
# ----------------------------------------------------------------------------
versoes = ["legado\n(-O3)", "CPU v1", "CPU v2", "CPU v3", "CUDA v1", "CUDA v2", "CUDA v3", "CUDA v4"]
treino = [0.3088, 0.1455, 0.0386, 0.0394, 3.6614, 0.9687, 0.4434, 0.2798]
wall = [0.3319, 0.1686, 0.0617, 0.0436, 5.7878, 3.0973, 2.5464, 2.3810]
cores = [COR["legado"]] + [COR["cpu"]] * 3 + [COR["cuda"]] * 4

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for ax, vals, titulo in ((ax1, treino, "Tempo de treino (s)"), (ax2, wall, "Tempo total de parede (s)")):
    barras = ax.bar(versoes, vals, color=cores)
    ax.set_yscale("log")
    ax.set_title(titulo)
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.4)
    for b, v in zip(barras, vals):
        ax.annotate(f"{v:.3g}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8)
    ax.axhline(vals[0], color="k", linestyle=":", alpha=0.5)
fig.suptitle("Barenn - Franky (no de GPU), mediana de 9 execucoes; todas as versoes com resultado identico bit a bit")
fig.tight_layout()
fig.savefig("docs/img/tempos_por_versao.png", dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------------
# 2) Speedup do treino sobre o legado -O3 (mesmo no)
# ----------------------------------------------------------------------------
speed = [treino[0] / t for t in treino]
fig, ax = plt.subplots(figsize=(9, 4.5))
barras = ax.bar(versoes, speed, color=cores)
ax.axhline(1.0, color="k", linestyle="--", alpha=0.6)
ax.set_yscale("log")
ax.set_ylabel("Speedup do treino sobre o legado -O3 (x)")
ax.set_title("Speedup de treino (>1 = mais rapido que o legado)")
ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.4)
for b, v in zip(barras, speed):
    ax.annotate(f"{v:.2f}x", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=9)
fig.tight_layout()
fig.savefig("docs/img/speedup_sobre_legado.png", dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------------
# 3) Ciclos por fase do passo de SGD na GPU (clock64, 1 bloco, 1 epoca)
# ----------------------------------------------------------------------------
v3 = [("forward L1", 1584), ("forward L2", 728), ("forward L3", 327), ("perda", 156),
      ("grad_in L3", 200), ("atualiza L3+d2", 330), ("grad_in L2", 287),
      ("atualiza L2+d1", 414), ("atualiza L1", 1108), ("commit prefetch", 98)]
v4 = [("forward L1", 532), ("forward L2", 276), ("forward L3", 211), ("A: grad_in L3 || perda", 213),
      ("B: atualiza L3 || grad_in L2", 464), ("C: atualiza L2 || d1", 201), ("D: atualiza L1 + prefetch", 494)]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, dados, titulo in ((axs[0], v3, "CUDA v3 - total 5232 ciclos/amostra"),
                          (axs[1], v4, "CUDA v4 - total 2391 ciclos/amostra")):
    nomes = [d[0] for d in dados][::-1]
    cic = [d[1] for d in dados][::-1]
    ax.barh(nomes, cic, color=COR["cuda"])
    ax.set_title(titulo)
    ax.set_xlabel("ciclos de GPU por amostra (clock64)")
    ax.set_xlim(0, 1750)
    for i, v in enumerate(cic):
        ax.annotate(str(v), (v, i), textcoords="offset points", xytext=(3, -3), fontsize=8)
fig.tight_layout()
fig.savefig("docs/img/ciclos_por_fase_gpu.png", dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------------
# 4) Escala do volume de dados (x100: 382.300 amostras de treino, 179.700 de teste)
# ----------------------------------------------------------------------------
nomes = ["legado", "CPU v2", "CPU v3", "CUDA v3", "CUDA v4"]
tr = [30.566, 3.7895, 3.8102, 40.744, 24.3996]
wl = [32.5055, 5.6231, 3.8569, 42.9308, 26.5580]
x = range(len(nomes))
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar([i - 0.2 for i in x], tr, 0.4, label="treino", color="#1f77b4")
ax.bar([i + 0.2 for i in x], wl, 0.4, label="tempo total (parede)", color="#ff7f0e")
ax.set_xticks(list(x))
ax.set_xticklabels(nomes)
ax.set_yscale("log")
ax.set_ylabel("segundos (escala log)")
ax.set_title("Dataset x100 (382 mil amostras): o treino escala linearmente em todas as versoes")
ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.4)
ax.legend()
for i, (a, b) in enumerate(zip(tr, wl)):
    ax.annotate(f"{a:.1f}", (i - 0.2, a), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.annotate(f"{b:.1f}", (i + 0.2, b), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
fig.tight_layout()
fig.savefig("docs/img/escala_x100.png", dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------------
# 5) OpenMP por dentro da amostra: o fork/join custa mais que o trabalho
# ----------------------------------------------------------------------------
rot = ["serial\n(CPU v2)", "OpenMP 2 thr\n(spin ativo)", "OpenMP 4 thr\n(spin ativo)",
       "OpenMP 2 thr\n(padrao)", "OpenMP 4 thr\n(padrao)"]
val = [0.0367, 0.4268, 0.5072, 1.4352, 1.6310]
fig, ax = plt.subplots(figsize=(9, 4.5))
barras = ax.bar(rot, val, color=["#1f77b4", "#d62728", "#d62728", "#8c1c1c", "#8c1c1c"])
ax.set_yscale("log")
ax.set_ylabel("Tempo de treino (s)")
ax.set_title("Experimento negativo: paralelizar POR DENTRO da amostra (camada 1) com OpenMP")
ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.4)
for b, v in zip(barras, val):
    ax.annotate(f"{v:.3g} s", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=9)
fig.tight_layout()
fig.savefig("docs/img/openmp_dentro_da_amostra.png", dpi=140)
plt.close(fig)
print("Graficos gerados em docs/img/")
