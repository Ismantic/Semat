ROOT      := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..
BUILD_DIR := $(ROOT)/build
PREPARE   := $(ROOT)/prepare
SCRIPTS   := $(ROOT)/scripts
OUTPUT    := $(SCRIPTS)/output/semat
CORPUS    := $(SCRIPTS)/News.dat.txt
VOCAB     := $(SCRIPTS)/count.txt
INIT      := $(PREPARE)/wavec.20260405.Kmeans.map
SEG_FILE  := $(PREPARE)/News.cut.txt
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

.PHONY: count conv fit print clean

count:
	python3 $(SCRIPTS)/process.py count $(SEG_FILE) $(VOCAB)

conv:
	@[ -f $(VOCAB) ] || $(MAKE) count
	python3 $(SCRIPTS)/process.py conv $(SEG_FILE) $(VOCAB) $(CORPUS) \
		--min-df $(MIN_DF) --min-len $(MIN_LEN) --min-score $(MIN_SCORE) --min-uniq $(MIN_UNIQ)

fit:
	@[ -f $(CORPUS) ] || $(MAKE) conv
	@mkdir -p $(dir $(OUTPUT))
	$(BUILD_DIR)/semat \
		$(CORPUS) $(TOPICS) $(ITERS) $(ALPHA) $(BETA) $(THREADS) \
		--init $(INIT) --output $(OUTPUT)

print:
	python3 $(SCRIPTS)/print_topics.py $(OUTPUT).phi $(TOPN)

clean:
	rm -rf $(SCRIPTS)/output
