// Barenn - CUDA v3: kernel persistente, pesos em memória compartilhada,
// prefetch da próxima amostra e avaliação na GPU com stream de cópia.
//
// Diagnóstico (nsys na v2): o passo já é um único kernel, mas ainda há 195 mil
// lançamentos (~1,2 us de piso cada), e a cada passo os pesos são relidos e
// reescritos na memória global.
// Técnicas (cada uma medida na Issue):
//   1. KERNEL PERSISTENTE: um único lançamento executa as 51 épocas; o laço
//      sobre as amostras roda DENTRO do kernel (zero lançamentos no meio).
//   2. PESOS EM MEMÓRIA COMPARTILHADA: os ~11 KB de pesos são carregados uma
//      vez e atualizados em shared memory (latência ~20 ciclos, sem tráfego
//      para L2); só voltam à global no fim.
//   3. PREFETCH COM DOUBLE BUFFERING: a próxima amostra é pedida à memória
//      global antes do cálculo da atual e gravada no buffer alternado ao final,
//      escondendo a latência (~500 ns) de uma leitura global por amostra.
//   4. AVALIAÇÃO NA GPU: um thread por imagem de teste (aqui sim há paralelismo
//      massivo), com a cópia do conjunto de teste feita numa stream separada,
//      sobreposta ao treino (memória "pinned" via cudaHostRegister).
//
// Resultado idêntico bit a bit ao do legado: a ordem de soma de cada neurônio
// é a mesma (um thread por neurônio, entradas em ordem crescente), compilado
// com --fmad=false.
#include "cuda_passo.cuh"
#include "leitura_rapida.hpp"

#ifndef TH
#define TH 128  // threads por bloco
#endif

using namespace std;
using namespace Barenn;

template <int T>
__global__ void __launch_bounds__(T)
k_treino(const float* __restrict__ imagens, const int* __restrict__ rotulos, int n,
         int epocas, float lr, float* gWt1, float* gb1, float* gWt2, float* gb2,
         float* gWt3, float* gb3, double* perdas_out) {
    __shared__ float sWt1[TAM_WT1], sb1[N1], sWt2[TAM_WT2], sb2[N2], sWt3[TAM_WT3], sb3[N3];
    __shared__ Compartilhado S;
    __shared__ float sx[2][N0];  // double buffer da amostra
    const int tid = threadIdx.x;

    for (int i = tid; i < TAM_WT1; i += T) sWt1[i] = gWt1[i];
    for (int i = tid; i < TAM_WT2; i += T) sWt2[i] = gWt2[i];
    for (int i = tid; i < TAM_WT3; i += T) sWt3[i] = gWt3[i];
    if (tid < N1) sb1[tid] = gb1[tid];
    if (tid < N2) sb2[tid] = gb2[tid];
    if (tid < N3) sb3[tid] = gb3[tid];

    for (int epoca = 0; epoca < epocas; ++epoca) {
        double acc = 0;  // só a thread 0 o altera
        if (tid < N0) sx[0][tid] = imagens[tid];
        int rot = rotulos[0];
        __syncthreads();
        for (int k = 0; k < n; ++k) {
            const int cur = k & 1;
            const bool tem_proxima = (k + 1 < n);
            // pede a próxima amostra agora; só é consumida no fim do passo
            float pre = 0;
            int prox_rot = 0;
            if (tem_proxima) {
                if (tid < N0) pre = imagens[static_cast<size_t>(k + 1) * N0 + tid];
                prox_rot = rotulos[k + 1];
            }
            passo_amostra<T>(sx[cur], rot, lr, sWt1, sb1, sWt2, sb2, sWt3, sb3, S, acc);
            if (tem_proxima) {
                if (tid < N0) sx[cur ^ 1][tid] = pre;
                rot = prox_rot;
            }
            __syncthreads();
        }
        if (tid == 0) perdas_out[epoca] = acc / n;
    }

    for (int i = tid; i < TAM_WT1; i += T) gWt1[i] = sWt1[i];
    for (int i = tid; i < TAM_WT2; i += T) gWt2[i] = sWt2[i];
    for (int i = tid; i < TAM_WT3; i += T) gWt3[i] = sWt3[i];
    if (tid < N1) gb1[tid] = sb1[tid];
    if (tid < N2) gb2[tid] = sb2[tid];
    if (tid < N3) gb3[tid] = sb3[tid];
}

// Um thread por imagem de teste. Mesma ordem de soma do legado.
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
        const Conjunto treino = ler_conjunto(arquivo_treino);
        const Conjunto teste = ler_conjunto(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }
        const int n = static_cast<int>(treino.size());
        const int nt = static_cast<int>(teste.size());
        const double t_carga = cron_carga.segundos();

        cudaStream_t s_treino, s_dados;
        CUDA_CHECK(cudaStreamCreate(&s_treino));
        CUDA_CHECK(cudaStreamCreate(&s_dados));
        // memória "pinned" permite cópia assíncrona de verdade
        const size_t bytes_teste = teste.imagens.size() * sizeof(float);
        CUDA_CHECK(cudaHostRegister(const_cast<float*>(teste.imagens.data()), bytes_teste,
                                    cudaHostRegisterDefault));

        // ---- Host -> Device (treino) ----
        Cronometro cron_h2d;
        PesosHost pesos = PesosHost::iniciais();
        PesosGpu g = empacotar(pesos);
        Dev<float> d_img(treino.imagens.size()), d_teste(teste.imagens.size());
        Dev<int> d_rot(treino.rotulos.size());
        Dev<float> d_wt1(TAM_WT1), d_b1(N1), d_wt2(TAM_WT2), d_b2(N2), d_wt3(TAM_WT3), d_b3(N3);
        Dev<double> d_perdas(epocas_efetivas());
        Dev<float> d_escores(static_cast<size_t>(nt) * N3);
        d_img.para_device(treino.imagens.data());
        d_rot.para_device(treino.rotulos.data());
        d_wt1.para_device(g.wt1.data()); d_b1.para_device(pesos.b1.data());
        d_wt2.para_device(g.wt2.data()); d_b2.para_device(pesos.b2.data());
        d_wt3.para_device(g.wt3.data()); d_b3.para_device(pesos.b3.data());
        const double t_h2d = cron_h2d.segundos();

        const int total_epocas = epocas_efetivas();

        // Informações do kernel (substitutas do Nsight Compute, indisponível aqui)
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, k_treino<TH>));
        int blocos_por_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocos_por_sm, k_treino<TH>, TH, 0));

        cudaEvent_t e0, e1, e2, e3, e_teste;
        CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventCreate(&e2)); CUDA_CHECK(cudaEventCreate(&e3));
        CUDA_CHECK(cudaEventCreate(&e_teste));

        // Cópia do conjunto de teste na stream de dados, sobreposta ao treino.
        CUDA_CHECK(cudaMemcpyAsync(d_teste.get(), teste.imagens.data(), bytes_teste,
                                   cudaMemcpyHostToDevice, s_dados));
        CUDA_CHECK(cudaEventRecord(e_teste, s_dados));

        // ---- treino: UM lançamento para todas as épocas ----
        CUDA_CHECK(cudaEventRecord(e0, s_treino));
        k_treino<TH><<<1, TH, 0, s_treino>>>(d_img.get(), d_rot.get(), n, total_epocas, taxa,
            d_wt1.get(), d_b1.get(), d_wt2.get(), d_b2.get(), d_wt3.get(), d_b3.get(),
            d_perdas.get());
        CUDA_CHECK(cudaEventRecord(e1, s_treino));
        CUDA_CHECK(cudaGetLastError());

        // ---- avaliação na GPU (espera a cópia do teste e o fim do treino) ----
        CUDA_CHECK(cudaStreamWaitEvent(s_treino, e_teste, 0));
        CUDA_CHECK(cudaEventRecord(e2, s_treino));
        k_avaliar<<<(nt + 127) / 128, 128, 0, s_treino>>>(d_teste.get(), nt,
            d_wt1.get(), d_b1.get(), d_wt2.get(), d_b2.get(), d_wt3.get(), d_b3.get(),
            d_escores.get());
        CUDA_CHECK(cudaEventRecord(e3, s_treino));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(s_treino));

        float ms_treino = 0, ms_aval = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_treino, e0, e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms_aval, e2, e3));
        const double t_treino = ms_treino / 1000.0;

        // ---- Device -> Host ----
        Cronometro cron_d2h;
        vector<double> perdas(total_epocas);
        vector<float> escores(static_cast<size_t>(nt) * N3);
        d_perdas.para_host(perdas.data());
        d_escores.para_host(escores.data());
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

        // A contagem final (acurácia, confusões, exemplos) reutiliza o avaliar()
        // do legado, alimentado com os escores calculados na GPU.
        Cronometro cron_aval;
        const float* base = teste.imagens.data();
        avaliar(teste, [&](const float* x, float* esc) {
            const size_t i = static_cast<size_t>(x - base) / N0;
            memcpy(esc, &escores[i * N3], N3 * sizeof(float));
        });
        imprimir_info_gpu();
        printf("INFO kernel_treino regs=%d smem_estatica=%zuB threads=%d blocos_por_SM=%d "
               "(ocupacao teorica %.1f%% das threads do SM)\n",
               attr.numRegs, attr.sharedSizeBytes, TH, blocos_por_sm,
               100.0 * blocos_por_sm * TH / 1536.0);
        printf("INFO avaliacao_gpu %.6f s (%d imagens)\n", ms_aval / 1000.0, nt);
        printf("INFO contexto_cuda %.6f s\nINFO h2d %.6f s\nINFO d2h %.6f s\n", t_ctx, t_h2d, t_d2h);
        printf("Tempo de carga %.6f s\nTempo de avaliacao %.6f s\n", t_carga, cron_aval.segundos());

        CUDA_CHECK(cudaHostUnregister(const_cast<float*>(teste.imagens.data())));
        CUDA_CHECK(cudaStreamDestroy(s_treino));
        CUDA_CHECK(cudaStreamDestroy(s_dados));
        CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
        CUDA_CHECK(cudaEventDestroy(e2)); CUDA_CHECK(cudaEventDestroy(e3));
        CUDA_CHECK(cudaEventDestroy(e_teste));
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
