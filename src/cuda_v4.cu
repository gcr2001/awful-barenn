// Barenn - CUDA v4: o passo de SGD reescrito para esconder latência.
//
// Diagnóstico (clock64 por fase, experimentos/cuda_fases.cu, na v3):
//   * forward camada 1 = 1584 ciclos para 64 somas dependentes (~25 ciclos por
//     termo; o limite real da cadeia de FADD seria ~4 ciclos por termo). SASS:
//     cada LDS é consumido logo pelo FMUL; com UM warp ativo não há quem
//     esconda os ~23 ciclos de latência da memória compartilhada.
//   * atualização da camada 1 = 1108 ciclos (índices idx/N, idx%N por elemento).
//   * 10 barreiras por passo, várias separando trabalho independente.
// Hipóteses aplicadas (nenhuma muda a ordem de soma de um neurônio, logo o
// resultado continua idêntico bit a bit ao do legado):
//   1. Os pesos do neurônio (coluna) vão para REGISTRADORES em lote antes das
//      somas: os LDS saem juntos, depois só FMUL/FADD (cadeia limitada por FADD).
//   2. O erro de cada classe é recalculado "inline" onde é usado, eliminando a
//      barreira entre perda e grad_in; tarefas independentes rodam em warps
//      diferentes (warp 0: atualiza camada 3 || warp 1: grad_in da camada 2;
//      perda no warp 1 || grad_in da camada 3 no warp 0). 10 -> 7 barreiras.
//   3. Atualizações com mapeamento fixo por thread (coluna k = tid & (largura-1)):
//      delta[k] fica em registrador e o laço é totalmente desenrolado.
//   4. x lido por float4 (broadcast de 16 B).
#include "cuda_avaliar.cuh"
#include "leitura_rapida.hpp"

#ifndef TH
#define TH 128  // threads por bloco (potência de 2 entre 64 e 512)
#endif
static_assert(TH >= 64 && TH <= 512 && (TH & (TH - 1)) == 0, "TH: potencia de 2 em [64,512]");

using namespace std;
using namespace Barenn;

struct alignas(16) Estado {
    alignas(16) float z1[N1];
    alignas(16) float h1[N1];
    alignas(16) float z2[N2];
    alignas(16) float h2[N2];
    alignas(16) float s[16];  // N3 = 10, acolchoado
    float e3[N2];             // grad_in da camada 3
    float d2[N2];             // delta da camada 2
    float gi2[N1];            // grad_in da camada 2
    float d1[N1];             // delta da camada 1
};

// sum + x[0]*w[0] + x[1]*w[1] + ...  (ordem crescente, uma soma por vez).
// `x` em memória compartilhada alinhada a 16 B; `w` em registradores.
template <int N>
__device__ __forceinline__ float soma_ordenada(float sum, const float* __restrict__ x,
                                               const float (&w)[N]) {
    #pragma unroll
    for (int q = 0; q < N / 4; ++q) {
        const float4 v = reinterpret_cast<const float4*>(x)[q];
        sum += v.x * w[4 * q + 0];
        sum += v.y * w[4 * q + 1];
        sum += v.z * w[4 * q + 2];
        sum += v.w * w[4 * q + 3];
    }
    return sum;
}

__device__ __forceinline__ float erro_de(const Estado& E, int i, int rotulo) {
    return E.s[i] - (i == rotulo ? 1.0f : 0.0f);
}

// Impede o compilador de mover leituras para depois das contas: força o padrão
// "todos os LDS em lote, depois FMUL/FADD/STS" (latência de memória compartilhada
// paga uma vez, não uma por elemento).
#define BARREIRA_COMPILADOR() asm volatile("" ::: "memory")

constexpr int TID_PERDA = 32;  // thread (warp 1) que soma a perda, em paralelo ao grad_in da camada 3

// minBlocksPerSM = 1: só UM bloco roda (o treino é sequencial), então o compilador
// pode usar até 255 registradores por thread em vez do teto heurístico (~63),
// que o fazia reaproveitar os mesmos registradores e serializar os elementos.
template <int T>
__global__ void __launch_bounds__(T, 1)
k_treino4(const float* __restrict__ imagens, const int* __restrict__ rotulos, int n,
          int epocas, float lr, float* gWt1, float* gb1, float* gWt2, float* gb2,
          float* gWt3, float* gb3, double* perdas_out) {
    __shared__ float sWt1[TAM_WT1], sb1[N1], sWt2[TAM_WT2], sb2[N2], sWt3[TAM_WT3], sb3[N3];
    __shared__ Estado E;
    __shared__ __align__(16) float sx[2][N0];
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;

    for (int i = tid; i < TAM_WT1; i += T) sWt1[i] = gWt1[i];
    for (int i = tid; i < TAM_WT2; i += T) sWt2[i] = gWt2[i];
    for (int i = tid; i < TAM_WT3; i += T) sWt3[i] = gWt3[i];
    if (tid < N1) sb1[tid] = gb1[tid];
    if (tid < N2) sb2[tid] = gb2[tid];
    if (tid < N3) sb3[tid] = gb3[tid];

    for (int epoca = 0; epoca < epocas; ++epoca) {
        double acc = 0;  // só a thread TID_PERDA o altera
        if (tid < N0) sx[0][tid] = imagens[tid];
        int rot = rotulos[0];
        __syncthreads();

        for (int k = 0; k < n; ++k) {
            const int cur = k & 1;
            const float* __restrict__ x = sx[cur];
            const bool tem_proxima = (k + 1 < n);
            float pre = 0;
            int prox_rot = 0;
            if (tem_proxima) {  // pedido agora, consumido no fim do passo
                if (tid < N0) pre = imagens[static_cast<size_t>(k + 1) * N0 + tid];
                prox_rot = rotulos[k + 1];
            }

            // ---- forward camada 1: warp 0, um neurônio por lane ----
            if (warp == 0) {
                float w[N0];
                #pragma unroll
                for (int j = 0; j < N0; ++j) w[j] = sWt1[j * P1 + lane];
                const float sum = soma_ordenada<N0>(sb1[lane], x, w);
                E.z1[lane] = sum;
                E.h1[lane] = relu_(sum);
            }
            __syncthreads();
            // ---- forward camada 2 ----
            if (tid < N2) {
                float w[N1];
                #pragma unroll
                for (int j = 0; j < N1; ++j) w[j] = sWt2[j * P2 + tid];
                const float sum = soma_ordenada<N1>(sb2[tid], E.h1, w);
                E.z2[tid] = sum;
                E.h2[tid] = relu_(sum);
            }
            __syncthreads();
            // ---- forward camada 3 (escores lineares) ----
            if (tid < N3) {
                float w[N2];
                #pragma unroll
                for (int j = 0; j < N2; ++j) w[j] = sWt3[j * P3 + tid];
                E.s[tid] = soma_ordenada<N2>(sb3[tid], E.h2, w);
            }
            __syncthreads();

            // ---- A: grad_in da camada 3 (warp 0)  ||  perda (warp 1) ----
            if (tid < N2) {
                float w[N3];
                #pragma unroll
                for (int j = 0; j < N3; ++j) w[j] = sWt3[tid * P3 + j];
                float sum = 0;
                #pragma unroll
                for (int j = 0; j < N3; ++j) sum += erro_de(E, j, rot) * w[j];
                E.e3[tid] = sum;
            }
            if (tid == TID_PERDA) {  // mesma ordem de soma do legado
                float total = 0;
                #pragma unroll
                for (int i = 0; i < N3; ++i) {
                    const float erro = erro_de(E, i, rot);
                    total += 0.5f * erro * erro;
                }
                acc += total;
            }
            __syncthreads();

            // ---- B: warp 0 atualiza camada 3 e calcula delta2 || warp 1: grad_in da camada 2 ----
            if (warp == 0) {
                if (lane < N2) E.d2[lane] = E.e3[lane] * (E.z2[lane] > 0.0f ? 1.0f : 0.0f);
                #pragma unroll
                for (int r = 0; r < (N2 * N3 + 31) / 32; ++r) {
                    const int idx = lane + 32 * r;
                    if (idx < N2 * N3) {
                        const int j = idx / N3, kk = idx % N3;
                        sWt3[j * P3 + kk] -= lr * (erro_de(E, kk, rot) * E.h2[j]);
                    }
                }
                if (lane < N3) sb3[lane] -= lr * erro_de(E, lane, rot);
            } else if (warp == 1) {
                float w[N2];
                #pragma unroll
                for (int j = 0; j < N2; ++j) w[j] = sWt2[lane * P2 + j];
                float sum = 0;
                #pragma unroll
                for (int j = 0; j < N2; ++j) {
                    const float dj = E.e3[j] * (E.z2[j] > 0.0f ? 1.0f : 0.0f);  // delta2[j]
                    sum += dj * w[j];
                }
                E.gi2[lane] = sum;
            }
            __syncthreads();

            // ---- C: todas atualizam a camada 2 || warp 0 calcula delta1 ----
            if (warp == 0) E.d1[lane] = E.gi2[lane] * (E.z1[lane] > 0.0f ? 1.0f : 0.0f);
            {
                const int kk = tid & 15, j0 = tid >> 4;
                constexpr int PASSO = T / 16, REP = N1 / PASSO;
                const float dk = E.d2[kk];
                float wv[REP], hv[REP];
                #pragma unroll
                for (int r = 0; r < REP; ++r) {  // todos os loads primeiro...
                    const int j = j0 + PASSO * r;
                    wv[r] = sWt2[j * P2 + kk];
                    hv[r] = E.h1[j];
                }
                BARREIRA_COMPILADOR();
                #pragma unroll
                for (int r = 0; r < REP; ++r) {  // ...depois contas e gravações
                    const int j = j0 + PASSO * r;
                    sWt2[j * P2 + kk] = wv[r] - lr * (dk * hv[r]);
                }
                if (tid < N2) sb2[tid] -= lr * E.d2[tid];
            }
            __syncthreads();

            // ---- D: todas atualizam a camada 1; confirma o prefetch ----
            {
                const int kk = tid & 31, j0 = tid >> 5;
                constexpr int PASSO = T / 32, REP = N0 / PASSO;
                const float dk = E.d1[kk];
                float wv[REP], xv[REP];
                #pragma unroll
                for (int r = 0; r < REP; ++r) {
                    const int j = j0 + PASSO * r;
                    wv[r] = sWt1[j * P1 + kk];
                    xv[r] = x[j];
                }
                BARREIRA_COMPILADOR();
                #pragma unroll
                for (int r = 0; r < REP; ++r) {
                    const int j = j0 + PASSO * r;
                    sWt1[j * P1 + kk] = wv[r] - lr * (dk * xv[r]);
                }
                if (tid < N1) sb1[tid] -= lr * E.d1[tid];
                if (tem_proxima) {
                    if (tid < N0) sx[cur ^ 1][tid] = pre;
                    rot = prox_rot;
                }
            }
            __syncthreads();
        }
        if (tid == TID_PERDA) perdas_out[epoca] = acc / n;
    }

    for (int i = tid; i < TAM_WT1; i += T) gWt1[i] = sWt1[i];
    for (int i = tid; i < TAM_WT2; i += T) gWt2[i] = sWt2[i];
    for (int i = tid; i < TAM_WT3; i += T) gWt3[i] = sWt3[i];
    if (tid < N1) gb1[tid] = sb1[tid];
    if (tid < N2) gb2[tid] = sb2[tid];
    if (tid < N3) gb3[tid] = sb3[tid];
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
        const size_t bytes_teste = teste.imagens.size() * sizeof(float);
        CUDA_CHECK(cudaHostRegister(const_cast<float*>(teste.imagens.data()), bytes_teste,
                                    cudaHostRegisterDefault));

        Cronometro cron_h2d;
        PesosHost pesos = PesosHost::iniciais();
        PesosGpu g = empacotar(pesos);
        const int total_epocas = epocas_efetivas();
        Dev<float> d_img(treino.imagens.size()), d_teste(teste.imagens.size());
        Dev<int> d_rot(treino.rotulos.size());
        Dev<float> d_wt1(TAM_WT1), d_b1(N1), d_wt2(TAM_WT2), d_b2(N2), d_wt3(TAM_WT3), d_b3(N3);
        Dev<double> d_perdas(total_epocas);
        Dev<float> d_escores(static_cast<size_t>(nt) * N3);
        d_img.para_device(treino.imagens.data());
        d_rot.para_device(treino.rotulos.data());
        d_wt1.para_device(g.wt1.data()); d_b1.para_device(pesos.b1.data());
        d_wt2.para_device(g.wt2.data()); d_b2.para_device(pesos.b2.data());
        d_wt3.para_device(g.wt3.data()); d_b3.para_device(pesos.b3.data());
        const double t_h2d = cron_h2d.segundos();

        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, k_treino4<TH>));
        int blocos_por_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocos_por_sm, k_treino4<TH>, TH, 0));

        cudaEvent_t e0, e1, e2, e3, e_teste;
        CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventCreate(&e2)); CUDA_CHECK(cudaEventCreate(&e3));
        CUDA_CHECK(cudaEventCreate(&e_teste));

        CUDA_CHECK(cudaMemcpyAsync(d_teste.get(), teste.imagens.data(), bytes_teste,
                                   cudaMemcpyHostToDevice, s_dados));
        CUDA_CHECK(cudaEventRecord(e_teste, s_dados));

        CUDA_CHECK(cudaEventRecord(e0, s_treino));
        k_treino4<TH><<<1, TH, 0, s_treino>>>(d_img.get(), d_rot.get(), n, total_epocas, taxa,
            d_wt1.get(), d_b1.get(), d_wt2.get(), d_b2.get(), d_wt3.get(), d_b3.get(),
            d_perdas.get());
        CUDA_CHECK(cudaEventRecord(e1, s_treino));
        CUDA_CHECK(cudaGetLastError());

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

        Cronometro cron_aval;
        const float* base = teste.imagens.data();
        avaliar(teste, [&](const float* x, float* esc) {
            const size_t i = static_cast<size_t>(x - base) / N0;
            memcpy(esc, &escores[i * N3], N3 * sizeof(float));
        });
        imprimir_info_gpu();
        printf("INFO kernel_treino regs=%d spill_local=%zuB smem_estatica=%zuB threads=%d "
               "blocos_por_SM=%d\n", attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes,
               TH, blocos_por_sm);
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
