// Barenn - CPU v3: multicore onde o paralelismo é EXATO.
//
// O treino é SGD amostra a amostra: os pesos usados na amostra t+1 dependem da
// atualização feita na amostra t, então o laço de treino NÃO pode ser dividido
// entre threads sem mudar o algoritmo (mini-batch / Hogwild produzem outro
// resultado). Paralelizar cada amostra por dentro também não compensa: o
// trabalho de uma amostra dura ~0,2 us e um fork/join de OpenMP custa mais que
// isso (medido em experimentos/cpu_omp_amostra.cpp e documentado na Issue).
//
// Fica então paralelo tudo o que é independente e produz resultado idêntico:
//   * leitura/interpretação das linhas dos arquivos (cada linha é independente);
//   * avaliação final (um forward por imagem; a contagem permanece sequencial
//     para preservar a ordem de soma da perda e os exemplos exibidos).
// O treino usa a mesma rede SIMD da v2.
#ifdef _OPENMP
#include <omp.h>
#endif

#include "leitura_rapida.hpp"
#include "rede_simd.hpp"

using namespace std;

static int num_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int main(int argc, char** argv) {
    string arquivo_treino, arquivo_teste;
    if (!Barenn::ler_argumentos(argc, argv, arquivo_treino, arquivo_teste)) {
        return 1;
    }

    try {
        Barenn::Cronometro cron_carga;
        const Barenn::Conjunto treino = Barenn::ler_conjunto(arquivo_treino);
        const Barenn::Conjunto teste = Barenn::ler_conjunto(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }
        const size_t n = treino.size();
        const double t_carga = cron_carga.segundos();

        auto rede = make_unique<Barenn::Rede>();
        array<double, Barenn::epocas> perdas{};

        Barenn::Cronometro cron_treino;
        for (int epoca = 0; epoca < Barenn::epocas; ++epoca) {
            double perda_total = 0;
            for (size_t k = 0; k < n; ++k) {
                perda_total += rede->passo(&treino.imagens[k * Barenn::entradas],
                                           treino.rotulos[k], Barenn::taxa);
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
            Barenn::Ativ a;  // local: forward é reentrante
            r.forward(x, a);
            memcpy(esc, a.s, Barenn::N3 * sizeof(float));
        }, /*paralelo=*/true);
        printf("INFO threads %d\nTempo de carga %.6f s\nTempo de avaliacao %.6f s\n",
               num_threads(), t_carga, cron_aval.segundos());
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
