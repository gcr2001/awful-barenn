// Teste de equivalência: ler_dados (legado) x ler_conjunto (rápido, paralelo).
//
// Para cada arquivo de entrada (válidos e inválidos), os dois leitores devem
// produzir exatamente os mesmos dados OU a mesma mensagem de erro (inclusive
// o número da linha). Compilar com: make test
#include <cstdio>

#include "../src/leitura_rapida.hpp"

using namespace std;

static string linha_valida(int rotulo = 3, const string& sufixo = "") {
    string s;
    for (int i = 0; i < Barenn::entradas; ++i) s += to_string(i % 17) + ",";
    return s + to_string(rotulo) + sufixo;
}

static void grava(const string& caminho, const string& conteudo) {
    ofstream f(caminho, ios::binary);
    f << conteudo;
}

struct Resultado {
    bool ok;
    string erro;
    Barenn::Conjunto dados;
};

template <class Leitor>
static Resultado ler(Leitor leitor, const string& caminho) {
    Resultado r{true, "", {}};
    try {
        r.dados = leitor(caminho);
    } catch (const exception& e) {
        r.ok = false;
        r.erro = e.what();
    }
    return r;
}

static bool iguais(const Resultado& a, const Resultado& b) {
    if (a.ok != b.ok || a.erro != b.erro) return false;
    if (a.dados.size() != b.dados.size()) return false;
    if (a.dados.rotulos != b.dados.rotulos) return false;
    // bit a bit: mesmos floats, na mesma ordem
    return a.dados.imagens.size() == b.dados.imagens.size() &&
           memcmp(a.dados.imagens.data(), b.dados.imagens.data(),
                  a.dados.imagens.size() * sizeof(float)) == 0;
}

int main() {
    const string v = linha_valida();
    struct Caso { const char* nome; string conteudo; };
    vector<Caso> casos = {
        {"valido, \\n final", v + "\n" + v + "\n"},
        {"valido, sem \\n final", v + "\n" + v},
        {"CRLF", v + "\r\n" + v + "\r\n"},
        {"espacos finais", v + "  \t\n"},
        {"rotulo com zero a esquerda", linha_valida(3, "") .substr(0, linha_valida().size() - 1) + "03\n"},
        {"atributo decimal", "1.5," + v.substr(v.find(',') + 1) + "\n"},
        {"atributo notacao cientifica", "1e1," + v.substr(v.find(',') + 1) + "\n"},
        {"atributo com sinal +", "+3," + v.substr(v.find(',') + 1) + "\n"},
        {"espaco antes da virgula", "1 ," + v.substr(v.find(',') + 1) + "\n"},
        {"atributo 17 (fora da faixa)", "17," + v.substr(v.find(',') + 1) + "\n"},
        {"atributo negativo", "-1," + v.substr(v.find(',') + 1) + "\n"},
        {"atributo com 3 digitos", "007," + v.substr(v.find(',') + 1) + "\n"},
        {"63 atributos", v.substr(v.find(',') + 1) + "\n"},
        {"65 atributos", "1," + v + "\n"},
        {"rotulo 10", linha_valida(10) + "\n"},
        {"rotulo negativo", linha_valida(-1) + "\n"},
        {"rotulo ausente", v.substr(0, v.size() - 1) + "\n"},
        {"lixo apos o rotulo", linha_valida(3, "x") + "\n"},
        {"linha vazia no meio", v + "\n\n" + v + "\n"},
        {"erro na 2a linha", v + "\nabc\n" + v + "\n"},
        {"duas linhas invalidas (reporta a 1a)", v + "\nabc\n" + v + "\n17,\n"},
        {"arquivo vazio", ""},
        {"so uma quebra de linha", "\n"},
    };

    int falhas = 0;
    for (const auto& c : casos) {
        grava("/tmp/barenn_teste.csv", c.conteudo);
        auto a = ler([](const string& p) { return Barenn::para_conjunto(Barenn::ler_dados(p)); }, "/tmp/barenn_teste.csv");
        auto b = ler([](const string& p) { return Barenn::ler_conjunto(p); }, "/tmp/barenn_teste.csv");
        const bool ok = iguais(a, b);
        printf("[%s] %-42s legado=%s rapido=%s\n", ok ? " OK " : "FALHA", c.nome,
               a.ok ? "dados" : "erro ", b.ok ? "dados" : "erro ");
        if (!ok) {
            ++falhas;
            if (a.erro != b.erro) printf("      legado: %s\n      rapido: %s\n", a.erro.c_str(), b.erro.c_str());
        }
    }

    // Arquivos reais do projeto.
    for (const char* arq : {"optdigits.tra", "optdigits.tes"}) {
        auto a = ler([](const string& p) { return Barenn::para_conjunto(Barenn::ler_dados(p)); }, arq);
        auto b = ler([](const string& p) { return Barenn::ler_conjunto(p); }, arq);
        const bool ok = iguais(a, b);
        printf("[%s] %-42s (%zu amostras)\n", ok ? " OK " : "FALHA", arq, a.dados.size());
        if (!ok) ++falhas;
    }
    remove("/tmp/barenn_teste.csv");
    printf("\n%s (%d falha(s))\n", falhas == 0 ? "TODOS OS TESTES PASSARAM" : "HA FALHAS", falhas);
    return falhas == 0 ? 0 : 1;
}
