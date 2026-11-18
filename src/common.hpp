// Código comum às versões otimizadas do Barenn (CPU e CUDA).
//
// A leitura dos dados, a função de perda e a avaliação final são cópias fiéis
// do código legado (barenn.cpp): o formato da saída é idêntico, o que permite
// comparar versões com `diff`. Alterações futuras neste arquivo devem ser
// apenas ADITIVAS, para não mudar o comportamento de versões já marcadas com
// tag.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Barenn {

using namespace std;

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;

// Hiperparâmetros do treino (iguais aos do legado).
constexpr int epocas = 51;
constexpr float taxa = 0.02f;

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

// Alocador que NÃO zera a memória em resize(): evita escrever 2x num bloco
// grande que será totalmente sobrescrito (e permite que o primeiro acesso às
// páginas seja feito em paralelo por quem as preenche).
template <class T>
struct SemInicializar : allocator<T> {
    template <class U>
    struct rebind {
        using other = SemInicializar<U>;
    };
    using allocator<T>::allocator;
    template <class U>
    void construct(U* p) noexcept { ::new (static_cast<void*>(p)) U; }
    template <class U, class... A>
    void construct(U* p, A&&... a) { ::new (static_cast<void*>(p)) U(std::forward<A>(a)...); }
};

// Conjunto de dados com as imagens em um único bloco contíguo (n x 64). É o
// formato usado pelo treino otimizado e pelo envio à GPU; também serve ao
// avaliar(), pois expõe operator[] com `.imagem.data()` e `.rotulo`.
struct Conjunto {
    vector<float, SemInicializar<float>> imagens;
    vector<int> rotulos;
    size_t size() const { return rotulos.size(); }

    struct Vista {
        struct Ptr {
            const float* p;
            const float* data() const { return p; }
        } imagem;
        int rotulo;
    };
    Vista operator[](size_t i) const {
        return {{imagens.data() + i * entradas}, rotulos[i]};
    }
};

inline Conjunto para_conjunto(const vector<Amostra>& amostras) {
    Conjunto c;
    c.imagens.resize(amostras.size() * entradas);
    c.rotulos.resize(amostras.size());
    for (size_t i = 0; i < amostras.size(); ++i) {
        copy(amostras[i].imagem.begin(), amostras[i].imagem.end(),
             c.imagens.begin() + i * entradas);
        c.rotulos[i] = amostras[i].rotulo;
    }
    return c;
}

// ---------------------------------------------------------------------------
// Leitura (idêntica ao legado)
// ---------------------------------------------------------------------------
inline vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    vector<Amostra> dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        Amostra amostra{};
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (int i = 0; i < entradas; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            amostra.imagem.push_back(valor / 16.0f);
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        if (amostra.imagem.size() != entradas ||
            !(campos >> amostra.rotulo) ||
            amostra.rotulo < 0 || amostra.rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.push_back(move(amostra));
    }
    if (arquivo.bad() || dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// ---------------------------------------------------------------------------
// Inicialização dos pesos (idêntica à do legado)
// ---------------------------------------------------------------------------
// Pesos de uma camada, ordem [saida][entrada]. Cada camada do legado recria o
// gerador com a MESMA semente (1337); reproduzir isso é indispensável para o
// resultado final ser idêntico.
inline vector<float> pesos_iniciais(int in, int out) {
    mt19937 gen(1337);
    uniform_real_distribution<float> distribution(-0.05f, 0.05f);
    vector<float> w(static_cast<size_t>(in) * out);
    for (float& x : w) x = distribution(gen);
    return w;
}

// ---------------------------------------------------------------------------
// Perda (mesma aritmética, mesma ordem de soma do legado)
// ---------------------------------------------------------------------------
// Metade da soma dos erros quadráticos; o erro também é a derivada por escore.
inline float perda(const float* escores, int rotulo, float* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            gradiente[i] = erro;
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// Avaliação final (mesma saída do legado)
// ---------------------------------------------------------------------------
// `forward(const float* imagem, float* escores)` preenche os 10 escores. Só é
// chamada para ler resultados: a fase 1 calcula todos os escores (pode ser
// paralela se `forward` for reentrante) e a fase 2 é a contagem sequencial,
// idêntica ao legado, que preserva a ordem de soma da perda e os exemplos
// exibidos.
template <class Dados, class Fwd>
void avaliar(const Dados& teste, Fwd&& forward, bool paralelo = false) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    const size_t n = teste.size();
    vector<array<float, classes>> todos(n);
    // Fase 1: um forward por imagem; independentes entre si.
    #pragma omp parallel for schedule(static) if (paralelo)
    for (long i = 0; i < static_cast<long>(n); ++i) {
        forward(teste[i].imagem.data(), todos[i].data());
    }

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    // Fase 2: contagem sequencial (ordem das imagens preservada).
    for (size_t i = 0; i < n; ++i) {
        const auto amostra = teste[i];
        const auto& escores = todos[i];
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores.data(), amostra.rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, amostra.rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

// ---------------------------------------------------------------------------
// Hash bit a bit (prova de equivalência numérica com o legado)
// ---------------------------------------------------------------------------
// FNV-1a de 64 bits. A ordem canônica dos pesos é a do legado: para cada
// camada, pesos [saida][entrada] e depois o viés.
struct Fnv {
    uint64_t h = 1469598103934665603ULL;
    void bytes(const void* p, size_t n) {
        const unsigned char* b = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < n; ++i) {
            h ^= b[i];
            h *= 1099511628211ULL;
        }
    }
    void mix(float v) { uint32_t b; memcpy(&b, &v, sizeof b); bytes(&b, 4); }
    void mix(double v) { uint64_t b; memcpy(&b, &v, sizeof b); bytes(&b, 8); }
};

// Imprime HASH_PESOS e HASH_PERDAS se a variável BARENN_HASH estiver definida.
inline void imprimir_hash_se_pedido(const Fnv& pesos, const double* perdas, int n) {
    if (!getenv("BARENN_HASH")) return;
    Fnv hl;
    for (int i = 0; i < n; ++i) hl.mix(perdas[i]);
    printf("HASH_PESOS %016llx\nHASH_PERDAS %016llx\n",
           static_cast<unsigned long long>(pesos.h),
           static_cast<unsigned long long>(hl.h));
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// Utilidades
// ---------------------------------------------------------------------------
struct Cronometro {
    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
    double segundos() const {
        return chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    }
    void reiniciar() { t0 = chrono::steady_clock::now(); }
};

// Mesmo contrato de argumentos do legado: nenhum, ou treino + teste.
inline bool ler_argumentos(int argc, char** argv, string& treino, string& teste) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return false;
    }
    treino = argc == 3 ? argv[1] : "optdigits.tra";
    teste = argc == 3 ? argv[2] : "optdigits.tes";
    return true;
}

}  // namespace Barenn
