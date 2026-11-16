// Experimento: quanto custa uma região paralela do OpenMP?
//
// Compara a latência de entrar e sair de `#pragma omp parallel` com o trabalho
// útil de UMA amostra do Barenn (~0,2 us na CPU v2). Se o fork/join custa mais
// que o trabalho que ele distribuiria, paralelizar por dentro da amostra só
// pode piorar. Compilar: g++ -O2 -fopenmp experimentos/omp_forkjoin.cpp
#include <omp.h>

#include <chrono>
#include <cstdio>
#include <initializer_list>

int main() {
    const int repeticoes = 200000;
    for (int threads : {1, 2, 4, 8}) {
        omp_set_num_threads(threads);
        // aquecimento: cria o time de threads
        #pragma omp parallel
        { asm volatile("" ::: "memory"); }

        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < repeticoes; ++i) {
            #pragma omp parallel
            { asm volatile("" ::: "memory"); }
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / repeticoes;
        printf("threads=%d  latencia de uma regiao paralela vazia: %.3f us\n", threads, us);
    }
    return 0;
}
