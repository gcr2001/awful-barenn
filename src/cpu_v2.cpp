// Barenn - CPU v2: layout SIMD com dimensões em tempo de compilação.
// A rede (Barenn::Rede) está em rede_simd.hpp; veja lá a explicação de por que
// o resultado é idêntico bit a bit ao do legado.
#include "rede_simd.hpp"

using namespace std;

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
        const Barenn::Conjunto dados = Barenn::para_conjunto(treino);
        const size_t n = dados.size();
        const double t_carga = cron_carga.segundos();

        auto rede = make_unique<Barenn::Rede>();
        array<double, Barenn::epocas> perdas{};

        Barenn::Cronometro cron_treino;
        for (int epoca = 0; epoca < Barenn::epocas; ++epoca) {
            double perda_total = 0;
            for (size_t k = 0; k < n; ++k) {
                perda_total += rede->passo(&dados.imagens[k * Barenn::entradas],
                                           dados.rotulos[k], Barenn::taxa);
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
        rede->hash_into(hash);
        Barenn::imprimir_hash_se_pedido(hash, perdas.data(), Barenn::epocas);

        Barenn::Cronometro cron_aval;
        const Barenn::Rede& r = *rede;
        Barenn::avaliar(teste, [&r](const float* x, float* esc) {
            Barenn::Ativ a;
            r.forward(x, a);
            memcpy(esc, a.s, Barenn::N3 * sizeof(float));
        });
        printf("Tempo de carga %.6f s\nTempo de avaliacao %.6f s\n",
               t_carga, cron_aval.segundos());
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
