ROOTDIR_SCALAR = ./opt/scalar
ROOTDIR_VEC = ./opt/vectorized
OPTS_SCALAR = opt1 opt2 opt3 opt4 opt5
OPTS_VEC = opt1 opt2 opt3 opt4

ALG1_SC = $(patsubst %,$(ROOTDIR_SCALAR)/%, $(OPTS_SCALAR))
ALG1_VEC = $(patsubst %,$(ROOTDIR_VEC)/%, $(OPTS_VEC))

all: $(ALG1_SC) $(ALG1_VEC)

$(ALG1_SC):
	$(MAKE) -C $@

$(ALG1_VEC):
	$(MAKE) -C $@

clean:
	for dir in $(ALG1_SC); do \
  		$(MAKE) -C $$dir -f Makefile $@; \
  done; \
  	for dir in $(ALG1_VEC); do \
  		$(MAKE) -C $$dir -f Makefile $@; \
  done;

.PHONY: $(ALG1_SC) $(ALG1_VEC)