ROOTDIR_SCALAR = ./opt/scalar
ROOTDIR_VEC = ./opt/vectorized
OPTS_SCALAR = opt1 opt2 opt3 opt4
OPTS_VEC = opt1 opt2 opt3 opt4

ALG2_SC = $(patsubst %,$(ROOTDIR_SCALAR)/%, $(OPTS_SCALAR))
ALG2_VEC = $(patsubst %,$(ROOTDIR_VEC)/%, $(OPTS_VEC))

all: $(ALG2_SC) $(ALG2_VEC)

$(ALG2_SC):
	$(MAKE) -C $@

$(ALG2_VEC):
	$(MAKE) -C $@

clean:
	for dir in $(ALG2_SC); do \
  		$(MAKE) -C $$dir -f Makefile $@; \
  done; \
  	for dir in $(ALG2_VEC); do \
  		$(MAKE) -C $$dir -f Makefile $@; \
  done;

.PHONY: $(ALG2_SC) $(ALG2_VEC)