ROOT      := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..
BUILD_DIR := $(ROOT)/build
PREPARE   := $(ROOT)/prepare
SCRIPTS   := $(ROOT)/scripts
WAVEC_ROOT ?= $(abspath $(ROOT)/../Wavec)
-include $(ROOT)/local.mk
NPROC     ?= 4
PY        ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,$(if $(wildcard $(ROOT)/.venv/bin/python),$(ROOT)/.venv/bin/python,python3))
RAW_CORPUS ?= $(ROOT)/data/derived/THUCNews.documents.txt
WAPIC_MODEL ?=
OUTPUT    ?= $(SCRIPTS)/output/semat
TRAIN_CORPUS ?= $(SCRIPTS)/News.dat.txt
VOCAB     ?= $(SCRIPTS)/count.txt
INIT      ?= $(WAVEC_ROOT)/data/wavec.20260405.Kmeans.map
SEG_FILE  ?= $(PREPARE)/THUCNews.cut.txt
TOPICS    ?= 100
ITERS     ?= 150
ALPHA     ?= 0.1
BETA      ?= 0.01
THREADS   ?= 8
MIN_DF    ?= 10
MIN_LEN   ?= 2
MIN_SCORE ?= 2.0
MIN_UNIQ  ?= 10
TOPN      ?= 30

.PHONY: all data build cut count conv wavec-init fit print test clean

all: print

data:
	$(MAKE) -C $(ROOT)/data all PY=$(PY)

build:
	cmake -S $(ROOT) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) -j$(NPROC)

$(RAW_CORPUS):
	$(MAKE) -C $(ROOT)/data all PY=$(PY)

cut: $(RAW_CORPUS)
	PYTHON=$(PY) bash $(SCRIPTS)/cut.sh $(RAW_CORPUS) $(SEG_FILE) $(NPROC) $(WAPIC_MODEL)

count:
	@[ -f $(SEG_FILE) ] || $(MAKE) cut
	$(PY) $(SCRIPTS)/process.py count $(SEG_FILE) $(VOCAB)

conv: wavec-init
	@[ -f $(VOCAB) ] || $(MAKE) count
	$(PY) $(SCRIPTS)/process.py conv $(SEG_FILE) $(VOCAB) $(TRAIN_CORPUS) \
		--min-df $(MIN_DF) --min-len $(MIN_LEN) --min-score $(MIN_SCORE) \
		--min-uniq $(MIN_UNIQ) --allowed-vocab $(INIT)

wavec-init:
	@test -f "$(INIT)" || { \
		echo "Missing Wavec cluster mapping: $(INIT)" >&2; \
		echo "Run Wavec filtering/K-means first, then set INIT to its clusters.map." >&2; \
		exit 1; \
	}

fit: build wavec-init
	@[ -f $(TRAIN_CORPUS) ] || $(MAKE) conv
	@mkdir -p $(dir $(OUTPUT))
	$(BUILD_DIR)/semat \
		$(TRAIN_CORPUS) $(TOPICS) $(ITERS) $(ALPHA) $(BETA) $(THREADS) \
		--init $(INIT) --output $(OUTPUT)

print:
	@[ -f $(OUTPUT).phi ] || $(MAKE) fit
	$(PY) $(SCRIPTS)/print_topics.py $(OUTPUT).phi $(TOPN)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	rm -rf $(SCRIPTS)/output
	rm -f $(SCRIPTS)/count.txt $(SCRIPTS)/News.dat.txt $(PREPARE)/THUCNews.cut.txt
