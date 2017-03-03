.PHONY: cuda
.PHONY: phi
.PHONY: data

Default:
	$(MAKE) cuda
	$(MAKE) phi


cuda:
	$(MAKE) -C cuda

phi:
	$(MAKE) -C phi

data:
	$(MAKE) -C data ARGS=${ARGS}
clean:
	rm cuda/cuda.out
