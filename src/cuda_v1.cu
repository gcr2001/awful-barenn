// Barenn - CUDA v1: port fiel e direto do algoritmo para a GPU.
//
// Objetivo desta versão: CORREÇÃO. Resultado idêntico bit a bit ao do legado
// (compilar com --fmad=false e -ffp-contract=off). Para isso cada neurônio é
// calculado por UMA thread, somando as entradas em ordem crescente, exatamente
// como o código sequencial; nenhuma redução em árvore (que reassociaria a soma).
//
// Estrutura (por amostra, 10 lançamentos de kernel numa única stream):
//   forward x3  ->  perda/gradiente  ->  backward da camada 3, 2 e 1  ->  avança
// Os dados de treino e os pesos ficam no device durante todo o treino; o host
// só enfileira kernels e sincroniza uma vez por época para ler a perda.
//
// Esta é a versão "ingênua": cada kernel dura ~1,2-2,9 us na GPU e o
// cudaLaunchKernel custa ~2,6 us na CPU (nsys), então o laço é limitado por
// lançamentos. As próximas versões atacam isso.
#include "cuda_util.cuh"

using namespace std;
using namespace Barenn;

__device__ __forceinline__ float relu(float s) { return (0.0f < s) ? s : 0.0f; }

// Gradiente em relação à soma antes da ativação (ReLU nas ocultas).
__device__ __forceinline__ float delta_de(const float* grad_out, const float* z, int i, bool final) {
    return final ? grad_out[i] : grad_out[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
}

// Um neurônio por thread. Se `cursor` != nullptr, a entrada é a amostra
// *cursor do conjunto de dados (stride floats por amostra).
__global__ void k_forward(const float* x, const int* cursor, int stride,
                          const float* W, const float* b, float* z, float* h,
                          int in, int out, bool final) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out) return;
    const float* xin = cursor ? x + static_cast<size_t>(*cursor) * stride : x;
    float sum = b[i];
    for (int j = 0; j < in; ++j) {
        sum += xin[j] * W[i * in + j];
    }
    z[i] = sum;
    h[i] = final ? sum : relu(sum);
}

// Perda da amostra e gradiente nos escores (mesma ordem de soma do legado).
__global__ void k_perda(const float* escores, const int* rotulos, const int* cursor,
                        float* grad, double* perda_acum) {
    const int rotulo = rotulos[*cursor];
    float total = 0;
    for (int i = 0; i < N3; ++i) {
        const float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        grad[i] = erro;
    }
    *perda_acum += total;
}

// grad_in[i] = sum_j delta[j] * W[j][i], j crescente, usando os pesos ANTES da
// atualização (este kernel roda antes de k_atualizar na mesma stream).
__global__ void k_grad_entrada(const float* grad_out, const float* z, const float* W,
                               float* grad_in, int in, int out, bool final) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= in) return;
    float sum = 0;
    for (int j = 0; j < out; ++j) {
        sum += delta_de(grad_out, z, j, final) * W[j * in + i];
    }
    grad_in[i] = sum;
}

// SGD: W[i][j] -= lr * (delta[i] * x[j]);  b[i] -= lr * delta[i].
__global__ void k_atualizar(const float* grad_out, const float* z, const float* x,
                            const int* cursor, int stride, float* W, float* b,
                            float lr, int in, int out, bool final) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = in * out;
    if (idx < total) {
        const int i = idx / in, j = idx % in;
        const float* xin = cursor ? x + static_cast<size_t>(*cursor) * stride : x;
        W[idx] -= lr * (delta_de(grad_out, z, i, final) * xin[j]);
    } else if (idx < total + out) {
        const int i = idx - total;
        b[i] -= lr * delta_de(grad_out, z, i, final);
    }
}

__global__ void k_avancar(int* cursor) { *cursor += 1; }

static int blocos(int n, int tpb) { return (n + tpb - 1) / tpb; }

int main(int argc, char** argv) {
    string arquivo_treino, arquivo_teste;
    if (!ler_argumentos(argc, argv, arquivo_treino, arquivo_teste)) {
        return 1;
    }

    try {
        Cronometro cron_ctx;
        CUDA_CHECK(cudaFree(0));  // força a criação do contexto fora da medição
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

        // ---- memória do device e transferências Host -> Device ----------------
        Cronometro cron_h2d;
        PesosHost pesos = PesosHost::iniciais();
        Dev<float> d_img(treino.imagens.size()), d_w1(pesos.w1.size()), d_b1(N1),
            d_w2(pesos.w2.size()), d_b2(N2), d_w3(pesos.w3.size()), d_b3(N3);
        Dev<int> d_rot(treino.rotulos.size()), d_cursor(1);
        Dev<float> d_z1(N1), d_h1(N1), d_z2(N2), d_h2(N2), d_z3(N3), d_s(N3);
        Dev<float> d_grad(N3), d_g3(N2), d_g2(N1);
        Dev<double> d_perda(1);
        d_img.para_device(treino.imagens.data());
        d_rot.para_device(treino.rotulos.data());
        d_w1.para_device(pesos.w1.data()); d_b1.para_device(pesos.b1.data());
        d_w2.para_device(pesos.w2.data()); d_b2.para_device(pesos.b2.data());
        d_w3.para_device(pesos.w3.data()); d_b3.para_device(pesos.b3.data());
        const double t_h2d = cron_h2d.segundos();

        const int total_epocas = epocas_efetivas();
        vector<double> perdas(total_epocas);

        Cronometro cron_treino;
        for (int epoca = 0; epoca < total_epocas; ++epoca) {
            d_cursor.zerar();
            d_perda.zerar();
            for (int k = 0; k < n; ++k) {
                k_forward<<<1, N1>>>(d_img.get(), d_cursor.get(), N0, d_w1.get(), d_b1.get(),
                                     d_z1.get(), d_h1.get(), N0, N1, false);
                k_forward<<<1, N2>>>(d_h1.get(), nullptr, 0, d_w2.get(), d_b2.get(),
                                     d_z2.get(), d_h2.get(), N1, N2, false);
                k_forward<<<1, N3>>>(d_h2.get(), nullptr, 0, d_w3.get(), d_b3.get(),
                                     d_z3.get(), d_s.get(), N2, N3, true);
                k_perda<<<1, 1>>>(d_s.get(), d_rot.get(), d_cursor.get(), d_grad.get(), d_perda.get());

                // camada 3 (final): grad_in com pesos antigos, depois atualização
                k_grad_entrada<<<1, N2>>>(d_grad.get(), d_z3.get(), d_w3.get(), d_g3.get(), N2, N3, true);
                k_atualizar<<<blocos(N2 * N3 + N3, 128), 128>>>(d_grad.get(), d_z3.get(), d_h2.get(),
                    nullptr, 0, d_w3.get(), d_b3.get(), taxa, N2, N3, true);
                // camada 2
                k_grad_entrada<<<1, N1>>>(d_g3.get(), d_z2.get(), d_w2.get(), d_g2.get(), N1, N2, false);
                k_atualizar<<<blocos(N1 * N2 + N2, 128), 128>>>(d_g3.get(), d_z2.get(), d_h1.get(),
                    nullptr, 0, d_w2.get(), d_b2.get(), taxa, N1, N2, false);
                // camada 1 (o grad_in da entrada nunca é usado)
                k_atualizar<<<blocos(N0 * N1 + N1, 128), 128>>>(d_g2.get(), d_z1.get(), d_img.get(),
                    d_cursor.get(), N0, d_w1.get(), d_b1.get(), taxa, N0, N1, false);
                k_avancar<<<1, 1>>>(d_cursor.get());
            }
            CUDA_CHECK(cudaGetLastError());
            double perda_total = 0;
            d_perda.para_host(&perda_total);  // também sincroniza a stream
            perdas[epoca] = perda_total / n;
        }
        const double t_treino = cron_treino.segundos();

        // ---- Device -> Host --------------------------------------------------
        Cronometro cron_d2h;
        d_w1.para_host(pesos.w1.data()); d_b1.para_host(pesos.b1.data());
        d_w2.para_host(pesos.w2.data()); d_b2.para_host(pesos.b2.data());
        d_w3.para_host(pesos.w3.data()); d_b3.para_host(pesos.b3.data());
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
        printf("INFO contexto_cuda %.6f s\nINFO h2d %.6f s\nINFO d2h %.6f s\n",
               t_ctx, t_h2d, t_d2h);
        printf("Tempo de carga %.6f s\nTempo de avaliacao %.6f s\n", t_carga, cron_aval.segundos());
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
