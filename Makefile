SUBDIRS = ./src ./benchmarking

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

.PHONY: $(SUBDIRS)