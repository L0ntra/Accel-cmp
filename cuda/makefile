Default:
	nvcc cuda.cu ../data/readdata.c -o cuda.out
	./cuda.out ../data/${ARGS}

test:
	./cuda.out ../data/testdata.txt 

rand:
	nvcc cuda_rand.cu -o cuda_rand.out
	./cuda_rand.out ${ARGS}

matmult:
	nvcc matrix_mult.cu -o matmult

clean:
	rm *.out
