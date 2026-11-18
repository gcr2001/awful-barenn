// Barenn - CPU v1: elimina trabalho morto, cópias e alocações.
//
// Mantém EXATAMENTE a aritmética e a ordem de soma do legado (resultado
// idêntico bit a bit). Mudanças, todas motivadas pelo profiling do legado:
//   * Backward da 1ª camada não calcula grad_in: o legado o calculava e o
//     descartava (75% do laço de grad_in era trabalho morto);
//   * Forward/Backward não recebem nem devolvem std::vector por valor e não
//     alocam memória: buffers pré-alocados e ponteiros __restrict;
//   * pesos num único bloco contíguo (antes: vector<vector<float>>);
//   * laço de grad_in reordenado (j externo, i interno): cada grad_in[i] ainda
//     soma delta[j]*w[j][i] em j crescente, só que agora o laço interno é
//     contíguo e vetorizável.
#include "common.hpp"

using namespace std;

namespace Barenn {

class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    vector<float> weights;  // weights[i*in_features + j]: entrada j -> saída i
    vector<float> bias;
    vector<float> last_z;
    vector<float> delta;
    const float* last_input = nullptr;  // aponta para buffer do chamador

public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(static_cast<size_t>(output_size) * input_size),
          bias(output_size, 0), last_z(output_size), delta(output_size) {
        // Mesma semente e mesma ordem de sorteio do legado.
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (float& weight : weights) {
            weight = distribution(gen);
        }
    }

    // `input` precisa permanecer válido até o Backward correspondente.
    void Forward(const float* __restrict input, float* __restrict out) {
        last_input = input;
        for (int i = 0; i < out_features; ++i) {
            const float* __restrict w = &weights[static_cast<size_t>(i) * in_features];
            float sum = bias[i];
            for (int j = 0; j < in_features; ++j) {
                sum += input[j] * w[j];
            }
            last_z[i] = sum;
            out[i] = final ? sum : max(0.0f, sum);
        }
    }

    // `grad_in` pode ser nullptr quando ninguém usa o gradiente da entrada.
    void Backward(const float* __restrict grad_out, float learning_rate,
                  float* __restrict grad_in) {
        for (int i = 0; i < out_features; ++i) {
            delta[i] = final ? grad_out[i]
                             : grad_out[i] * (last_z[i] > 0.0f ? 1.0f : 0.0f);
        }

        // Usa os pesos anteriores à atualização, como no legado.
        if (grad_in) {
            fill(grad_in, grad_in + in_features, 0.0f);
            for (int j = 0; j < out_features; ++j) {
                const float dj = delta[j];
                const float* __restrict w = &weights[static_cast<size_t>(j) * in_features];
                for (int i = 0; i < in_features; ++i) {
                    grad_in[i] += dj * w[i];
                }
            }
        }

        const float* __restrict x = last_input;
        for (int i = 0; i < out_features; ++i) {
            const float di = delta[i];
            float* __restrict w = &weights[static_cast<size_t>(i) * in_features];
            for (int j = 0; j < in_features; ++j) {
                w[j] -= learning_rate * (di * x[j]);
            }
            bias[i] -= learning_rate * di;
        }
    }

    void HashInto(Fnv& f) const {
        for (float w : weights) f.mix(w);
        for (float b : bias) f.mix(b);
    }
};

}  // namespace Barenn

int main(int argc, char** argv) {
    string arquivo_treino, arquivo_teste;
    if (!Barenn::ler_argumentos(argc, argv, arquivo_treino, arquivo_teste)) {
        return 1;
    }

    try {
        Barenn::Cronometro cron_carga;
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }
        // Imagens em um único bloco contíguo (melhor localidade que vector<vector>).
        const Barenn::Conjunto dados = Barenn::para_conjunto(treino);
        const size_t n = dados.size();
        const double t_carga = cron_carga.segundos();

        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        float h1[32], h2[16], escores[Barenn::classes];
        float gradiente[Barenn::classes], g3[16], g2[32];
        array<double, Barenn::epocas> perdas{};

        Barenn::Cronometro cron_treino;
        for (int epoca = 0; epoca < Barenn::epocas; ++epoca) {
            double perda_total = 0;
            for (size_t k = 0; k < n; ++k) {
                const float* x = &dados.imagens[k * Barenn::entradas];
                l1.Forward(x, h1);
                l2.Forward(h1, h2);
                l3.Forward(h2, escores);
                perda_total += Barenn::perda(escores, dados.rotulos[k], gradiente);

                l3.Backward(gradiente, Barenn::taxa, g3);
                l2.Backward(g3, Barenn::taxa, g2);
                l1.Backward(g2, Barenn::taxa, nullptr);  // grad_in não é usado
            }
            perdas[epoca] = perda_total / n;
        }
        const double t_treino = cron_treino.segundos();

        cout << fixed << setprecision(4)
             << "TREINO | " << n << " imagens\n";
        for (int epoca = 0; epoca < Barenn::epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << Barenn::epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << t_treino << " s\n";

        Barenn::Fnv hash;
        l1.HashInto(hash);
        l2.HashInto(hash);
        l3.HashInto(hash);
        Barenn::imprimir_hash_se_pedido(hash, perdas.data(), Barenn::epocas);

        Barenn::Cronometro cron_aval;
        Barenn::avaliar(teste, [&](const float* x, float* esc) {
            float a[32], b[16];
            l1.Forward(x, a);
            l2.Forward(a, b);
            l3.Forward(b, esc);
        });
        printf("Tempo de carga %.6f s\nTempo de avaliacao %.6f s\n",
               t_carga, cron_aval.segundos());
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
