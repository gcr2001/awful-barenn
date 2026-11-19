// Leitura rápida e paralela dos dados (CPU v3 em diante).
//
// Header separado de common.hpp para que cada bloco de trabalho só ADICIONE
// arquivos: as versões anteriores continuam compilando exatamente como estavam.
//
// Estratégia: (1) lê o arquivo inteiro de uma vez; (2) separa as linhas;
// (3) cada linha é interpretada de forma independente (OpenMP), escrevendo
// direto no bloco contíguo do Conjunto (sem alocação por linha).
// O caminho rápido só aceita o formato canônico (64 inteiros 0..99 seguidos
// de vírgula e um rótulo de um dígito); qualquer outra coisa cai no mesmo
// parser baseado em istringstream do legado, então a SEMÂNTICA é a mesma para
// qualquer entrada (inclusive os erros: mesma mensagem, mesma linha).
#pragma once
#include <memory>

#include "common.hpp"

namespace Barenn {

// Interpreta uma linha com a lógica exata do legado. false = linha inválida.
inline bool parse_linha_legado(const string& linha, Amostra& amostra) {
    istringstream campos(linha);
    amostra.imagem.clear();
    for (int i = 0; i < entradas; ++i) {
        float valor;
        char separador;
        if (!(campos >> valor >> separador) || separador != ',' ||
            !isfinite(valor) || valor < 0 || valor > 16) {
            break;
        }
        amostra.imagem.push_back(valor / 16.0f);
    }
    return amostra.imagem.size() == entradas &&
           (campos >> amostra.rotulo) &&
           amostra.rotulo >= 0 && amostra.rotulo < classes &&
           (campos >> ws).eof();
}

// Caminho rápido: escreve os 64 atributos em `saida`. false = formato fora do
// canônico (o chamador usa o parser do legado).
inline bool parse_linha_rapido(const char* p, const char* fim, float* saida, int& rotulo) {
    for (int i = 0; i < entradas; ++i) {
        if (p >= fim || *p < '0' || *p > '9') return false;
        int v = *p++ - '0';
        if (p < fim && *p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
        if (p >= fim || *p != ',') return false;  // 3+ dígitos, espaço, '.', ...
        ++p;
        if (v > 16) return false;
        saida[i] = static_cast<float>(v) / 16.0f;
    }
    if (p >= fim || *p < '0' || *p > '9') return false;
    const int r = *p++ - '0';
    if (p < fim && *p >= '0' && *p <= '9') return false;  // 2+ dígitos: legado decide
    while (p < fim) {  // só espaço em branco até o fim da linha (como `>> ws`)
        const char c = *p++;
        if (c != ' ' && c != '\t' && c != '\r' && c != '\v' && c != '\f') return false;
    }
    rotulo = r;
    return true;
}

// Equivalente a para_conjunto(ler_dados(caminho)), muito mais rápido.
inline Conjunto ler_conjunto(const string& caminho) {
    ifstream arquivo(caminho, ios::binary);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }
    arquivo.seekg(0, ios::end);
    const streamoff tamanho = arquivo.tellg();
    if (tamanho < 0) {  // não é um arquivo comum: comportamento do legado
        return para_conjunto(ler_dados(caminho));
    }
    arquivo.seekg(0, ios::beg);
    unique_ptr<char[]> conteudo(new char[static_cast<size_t>(tamanho) + 1]);  // sem zerar
    if (tamanho > 0) arquivo.read(conteudo.get(), tamanho);
    if (arquivo.bad() || (tamanho > 0 && arquivo.gcount() != tamanho)) {
        return para_conjunto(ler_dados(caminho));
    }

    // Limites de cada linha, como getline: separador '\n'; sem linha extra
    // vazia depois de um '\n' final.
    vector<pair<size_t, size_t>> linhas;
    linhas.reserve(static_cast<size_t>(tamanho) / 100 + 16);
    for (size_t pos = 0; pos < static_cast<size_t>(tamanho);) {
        const void* nl = memchr(conteudo.get() + pos, '\n', tamanho - pos);
        const size_t fim = nl ? static_cast<const char*>(nl) - conteudo.get()
                              : static_cast<size_t>(tamanho);
        linhas.emplace_back(pos, fim);
        pos = fim + 1;
    }
    if (linhas.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }

    const long n = static_cast<long>(linhas.size());
    Conjunto c;
    c.imagens.resize(static_cast<size_t>(n) * entradas);  // sem zerar
    c.rotulos.resize(static_cast<size_t>(n));
    vector<char> invalida(static_cast<size_t>(n), 0);
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {
        const char* ini = conteudo.get() + linhas[i].first;
        const char* fim = conteudo.get() + linhas[i].second;
        float* saida = c.imagens.data() + static_cast<size_t>(i) * entradas;
        int rotulo = 0;
        if (!parse_linha_rapido(ini, fim, saida, rotulo)) {
            Amostra tmp{};
            if (parse_linha_legado(string(ini, fim), tmp)) {
                copy(tmp.imagem.begin(), tmp.imagem.end(), saida);
                rotulo = tmp.rotulo;
            } else {
                invalida[i] = 1;
            }
        }
        c.rotulos[i] = rotulo;
    }
    // Primeira linha inválida (a de menor número), como no laço sequencial.
    for (long i = 0; i < n; ++i) {
        if (invalida[i]) {
            throw runtime_error(caminho + ":" + to_string(i + 1) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
    }
    return c;
}

}  // namespace Barenn
