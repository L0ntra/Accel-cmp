Default:
	nvcc cuda.cu ../data/readdata.c -o cuda.out
	./cuda.out ../data/${ARGS}

test:
	./cuda.out ../data/testdata.txt 

clean:
	rm *.out
