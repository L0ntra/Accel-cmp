Default:
	$(MAKE) cuda
	$(MAKE) phi

cuda:
	nvcc cuda/cuda.cu -o cuda/cuda.out
	./cuda/cuda.out 2


phi:
	ls


clean:
	rm cuda/cuda.out
