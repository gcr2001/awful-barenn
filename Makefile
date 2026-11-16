# Barenn - build reprodutivel.
#
#   make              compila tudo que existir em src/ (e o legado)
#   make bin/cpu_v2   compila uma versao especifica
#   make ISA=-mavx2   (GCC antigo, p.ex. SDumont/GCC 8: sem -march=x86-64-v3)
#   make ARCH=sm_70   (CUDA para V100 do SDumont; padrao sm_86 = RTX A1000)
#
# Politica numerica: SEMPRE -ffp-contract=off. Sem isso o compilador funde
# a*b+c em FMA e os pesos finais deixam de ser identicos aos do legado
# (comprovado nas Issues). Nunca usar -ffast-math/-Ofast.
CXX      ?= g++
NVCC     ?= /usr/local/cuda/bin/nvcc
STD      := -std=c++17
ISA      ?= -march=x86-64-v3
ARCH     ?= sm_86
FP       := -ffp-contract=off
# Os nos de computacao nao carregam modulos: embute o caminho da libstdc++.
RPATH    := -Wl,-rpath,$(shell dirname $$($(CXX) -print-file-name=libstdc++.so))

CPU_SRC  := $(wildcard src/cpu_v*.cpp)
CUDA_SRC := $(wildcard src/cuda_v*.cu)
CPU_BIN  := $(patsubst src/%.cpp,bin/%,$(CPU_SRC))
TEST_SRC := $(wildcard tests/test_*.cpp)
TEST_BIN := $(patsubst tests/%.cpp,bin/%,$(TEST_SRC))
CUDA_BIN := $(patsubst src/%.cu,bin/%,$(CUDA_SRC))
HAS_NVCC := $(shell test -x $(NVCC) && echo yes)

all: bin/legado bin/legado_v3 $(CPU_BIN) $(if $(HAS_NVCC),$(CUDA_BIN))

bin:
	mkdir -p bin

# Legado, compilado de forma "pura" (x86-64 generico) e com a mesma ISA das
# versoes otimizadas (comparacao justa de flags).
bin/legado: barenn.cpp | bin
	$(CXX) $(STD) -O3 $(FP) $(RPATH) $< -o $@
bin/legado_v3: barenn.cpp | bin
	$(CXX) $(STD) -O3 $(ISA) $(FP) $(RPATH) $< -o $@

# Versoes com OpenMP (multicore) levam -fopenmp. (Nota: o '%' do make nao casa
# com string vazia, por isso listamos os binarios explicitamente.)
OMP_BINS := bin/cpu_v3
$(OMP_BINS): EXTRA := -fopenmp
bin/cpu_v%: src/cpu_v%.cpp src/common.hpp | bin
	$(CXX) $(STD) -O3 $(ISA) $(FP) $(EXTRA) $(RPATH) $< -o $@

# cuda_v3 e cuda_v4 reutilizam a leitura paralela (OpenMP no host).
bin/cuda_v3 bin/cuda_v4: NVEXTRA := -Xcompiler -fopenmp -lgomp
bin/cuda_v%: src/cuda_v%.cu $(wildcard src/*.cuh) $(wildcard src/*.hpp) | bin
	$(NVCC) $(STD) -O3 -arch=$(ARCH) --fmad=false -Xcompiler -ffp-contract=off $(NVEXTRA) \
	    -Xlinker -rpath=$(shell dirname $$($(CXX) -print-file-name=libstdc++.so)) $< -o $@

# Testes de equivalencia (make test)
bin/test_%: tests/test_%.cpp $(wildcard src/*.hpp) | bin
	$(CXX) $(STD) -O2 -fopenmp $(RPATH) $< -o $@

test: $(TEST_BIN)
	@for t in $(TEST_BIN); do echo "== $$t =="; ./$$t || exit 1; done

clean:
	rm -rf bin build

.PHONY: all clean test
