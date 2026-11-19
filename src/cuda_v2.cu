// Barenn - CUDA v2: um kernel fundido por amostra (10 lançamentos -> 1).
//
// Diagnóstico (nsys na v1): cada kernel dura 1,2-2,9 us, cudaLaunchKernel custa
// ~2,6 us na CPU e há ~27% de ociosidade entre lançamentos. Reduzir só o custo
// de lançamento (CUDA Graphs) não basta: o piso de ~1,2 us POR kernel continua
// valendo para cada um dos 10. Previsão: <= ~1,3x. Medido num experimento
// (experimentos/gerar_cuda_v1_graph.py): 3,66 s -> 2,68 s = 1,37x.
// Hipótese: fundir o passo inteiro em um só kernel, com as camadas separadas por
// __syncthreads() (barreira de bloco, ~25 ciclos) em vez de por fronteiras de
// kernel (~1,2 us).
//
// O passo roda num único bloco (ver cuda_passo.cuh); ativações e gradientes em
// memória compartilhada, pesos em memória global (layout transposto, pitch
// ímpar). Resultado idêntico bit a bit ao do legado.
#include "cuda_passo.cuh"

#ifndef TH
#define TH 128  // threads por bloco
#endif

using namespace std;
using namespace Barenn;

__global__ void k_passo(const float* __restrict__ x, int rotulo, float lr,
                        float* Wt1, float* b1, float* Wt2, float* b2,
                        float* Wt3, float* b3, double* perda) {
    __shared__ Compartilhado S;
    double acc = 0;
    if (threadIdx.x == 0) acc = *perda;
    passo_amostra<TH>(x, rotulo, lr, Wt1, b1, Wt2, b2, Wt3, b3, S, acc);
    if (threadIdx.x == 0) *perda = acc;
}

int main(int argc, char** argv) {
    string arquivo_treino, arquivo_teste;
    if (!ler_argumentos(argc, argv, arquivo_treino, arquivo_teste)) {
        return 1;
    }

    try {
        Cronometro cron_ctx;
        CUDA_CHECK(cudaFree(0));
        const double t_ctx = cron_ctx.segundos();

        Cronometro cron_carga;
        const auto treino_amostras = ler_dados(arquivo_treino);
        const auto teste = ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }
        const Conjunto treino = para_conjunto(treino_amostras);
        const int n = static_cast<int>(treino.size());
        const double t_carga = cron_carga.segundos();

        // ---- Host -> Device ----
        Cronometro cron_h2d;
        PesosHost pesos = PesosHost::iniciais();
        PesosGpu g = empacotar(pesos);
        Dev<float> d_img(treino.imagens.size());
        Dev<float> d_wt1(TAM_WT1), d_b1(N1), d_wt2(TAM_WT2), d_b2(N2), d_wt3(TAM_WT3), d_b3(N3);
        Dev<double> d_perda(1);
        d_img.para_device(treino.imagens.data());
        d_wt1.para_device(g.wt1.data()); d_b1.para_device(pesos.b1.data());
        d_wt2.para_device(g.wt2.data()); d_b2.para_device(pesos.b2.data());
        d_wt3.para_device(g.wt3.data()); d_b3.para_device(pesos.b3.data());
        const double t_h2d = cron_h2d.segundos();

        const int total_epocas = epocas_efetivas();
        vector<double> perdas(total_epocas);

        Cronometro cron_treino;
        for (int epoca = 0; epoca < total_epocas; ++epoca) {
            d_perda.zerar();
            for (int k = 0; k < n; ++k) {
                k_passo<<<1, TH>>>(d_img.get() + static_cast<size_t>(k) * N0, treino.rotulos[k], taxa,
                                   d_wt1.get(), d_b1.get(), d_wt2.get(), d_b2.get(),
                                   d_wt3.get(), d_b3.get(), d_perda.get());
            }
            CUDA_CHECK(cudaGetLastError());
            double perda_total = 0;
            d_perda.para_host(&perda_total);
            perdas[epoca] = perda_total / n;
        }
        const double t_treino = cron_treino.segundos();

        // ---- Device -> Host ----
        Cronometro cron_d2h;
        d_wt1.para_host(g.wt1.data()); d_b1.para_host(pesos.b1.data());
        d_wt2.para_host(g.wt2.data()); d_b2.para_host(pesos.b2.data());
        d_wt3.para_host(g.wt3.data()); d_b3.para_host(pesos.b3.data());
        desempacotar(g, pesos);
        const double t_d2h = cron_d2h.segundos();

        cout << fixed << setprecision(4) << "TREINO | " << n << " imagens\n";
        for (int epoca = 0; epoca < total_epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << total_epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << t_treino << " s\n";

        Fnv hash;
        pesos.hash_into(hash);
        imprimir_hash_se_pedido(hash, perdas.data(), total_epocas);

        Cronometro cron_aval;
        avaliar(teste, [&pesos](const float* x, float* esc) { pesos.forward(x, esc); });
        imprimir_info_gpu();
        printf("INFO threads_por_bloco %d\nINFO contexto_cuda %.6f s\nINFO h2d %.6f s\nINFO d2h %.6f s\n",
               TH, t_ctx, t_h2d, t_d2h);
        printf("Tempo de carga %.6f s\nTempo de avaliacao %.6f s\n", t_carga, cron_aval.segundos());
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
