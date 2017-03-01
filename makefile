.PHONY: cuda
.PHONY: phi

Default:
	$(MAKE) cuda
	$(MAKE) phi


cuda:
	$(MAKE) -C cuda

phi:
	$(MAKE) -C phi

clean:
	rm cuda/cuda.out
