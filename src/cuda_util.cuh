// Utilitários das versões CUDA: checagem de erros, memória do device (RAII),
// pesos no host em layout canônico do legado e avaliação final no host.
//
// Convenção de layout em TODAS as versões CUDA ao trocar dados com o host:
// pesos [saida][entrada] por camada, exatamente como no legado. Assim o hash
// dos pesos e a avaliação final são idênticos aos das versões de CPU.
#pragma once

#include <cuda_runtime.h>

#include "common.hpp"

namespace Barenn {

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t erro_cuda_ = (call);                                          \
        if (erro_cuda_ != cudaSuccess) {                                          \
            throw std::runtime_error(std::string("CUDA: ") +                      \
                cudaGetErrorString(erro_cuda_) + " em " #call " (" __FILE__ ":" + \
                std::to_string(__LINE__) + ")");                                  \
        }                                                                         \
    } while (0)

// Dimensões fixas da rede 64 -> 32 -> 16 -> 10.
constexpr int N0 = entradas;
constexpr int N1 = 32;
constexpr int N2 = 16;
constexpr int N3 = classes;

// Bloco de memória do device com liberação automática (RAII).
template <class T>
class Dev {
    T* p_ = nullptr;
    size_t n_ = 0;

public:
    Dev() = default;
    explicit Dev(size_t n) { alocar(n); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    ~Dev() { if (p_) cudaFree(p_); }
    void alocar(size_t n) {
        if (p_) cudaFree(p_);
        n_ = n;
        CUDA_CHECK(cudaMalloc(&p_, n * sizeof(T)));
    }
    T* get() const { return p_; }
    size_t size() const { return n_; }
    void zerar() { CUDA_CHECK(cudaMemset(p_, 0, n_ * sizeof(T))); }
    void para_device(const T* host) {
        CUDA_CHECK(cudaMemcpy(p_, host, n_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    void para_host(T* host) const {
        CUDA_CHECK(cudaMemcpy(host, p_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

// Pesos e vieses das três camadas, no layout do legado ([saida][entrada]).
struct PesosHost {
    vector<float> w1, b1, w2, b2, w3, b3;

    static PesosHost iniciais() {
        PesosHost p;
        p.w1 = pesos_iniciais(N0, N1);
        p.w2 = pesos_iniciais(N1, N2);
        p.w3 = pesos_iniciais(N2, N3);
        p.b1.assign(N1, 0.0f);
        p.b2.assign(N2, 0.0f);
        p.b3.assign(N3, 0.0f);
        return p;
    }

    // Ordem canônica do legado: por camada, pesos e depois viés.
    void hash_into(Fnv& f) const {
        for (float v : w1) f.mix(v);
        for (float v : b1) f.mix(v);
        for (float v : w2) f.mix(v);
        for (float v : b2) f.mix(v);
        for (float v : w3) f.mix(v);
        for (float v : b3) f.mix(v);
    }

    // Forward escalar com a MESMA ordem de soma do legado (sem FMA).
    static void camada(const float* in, int ni, const float* w, const float* b,
                       int no, bool final, float* out) {
        for (int i = 0; i < no; ++i) {
            float sum = b[i];
            for (int j = 0; j < ni; ++j) {
                sum += in[j] * w[static_cast<size_t>(i) * ni + j];
            }
            out[i] = final ? sum : ((0.0f < sum) ? sum : 0.0f);
        }
    }
    void forward(const float* x, float* escores) const {
        float h1[N1], h2[N2];
        camada(x, N0, w1.data(), b1.data(), N1, false, h1);
        camada(h1, N1, w2.data(), b2.data(), N2, false, h2);
        camada(h2, N2, w3.data(), b3.data(), N3, true, escores);
    }
};

// Número de épocas efetivo. A variável BARENN_EPOCAS só existe para permitir
// profiling curto (p.ex. nsys com 1-2 épocas); sem ela, valem as 51 do legado.
inline int epocas_efetivas() {
    if (const char* e = getenv("BARENN_EPOCAS")) {
        const int v = atoi(e);
        if (v > 0) return v;
    }
    return epocas;
}

// Informações do dispositivo para registro nos relatórios.
inline void imprimir_info_gpu() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("INFO gpu \"%s\" sm_%d%d SMs=%d clock=%.0fMHz\n", p.name, p.major, p.minor,
           p.multiProcessorCount, p.clockRate / 1000.0);
}

}  // namespace Barenn
