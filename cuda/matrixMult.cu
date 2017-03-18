#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define TILE_WIDTH 32

void printM(float *M, int Width) {
	for(int i = 0; i < Width; ++i) {
		for(int j = 0; j < Width; ++j) {
			printf("%f ",M[Width*i+j]);
		}	
		printf("\n");
	}
}

__global__ 
void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width) {
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;int by = blockIdx.y;
	int tx = threadIdx.x;int ty = threadIdx.y;

	int Row = by*TILE_WIDTH + ty;
	int Col = bx*TILE_WIDTH + tx;

	float Pvalue= 0;
	for (int ph = 0; ph <(int) ceil(Width/(float)TILE_WIDTH); ++ph) {
		if((Row < Width) && (ph*TILE_WIDTH + tx)< Width)
			Mds[ty][tx] = d_M[Row*Width + (ph*TILE_WIDTH + tx)];
		else // Zero out extra threads that don't cover matrices
			Mds[ty][tx] = 0.0;
		if((Col < Width) && (ph*TILE_WIDTH + ty)< Width)
			Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty)*Width + Col];
		else
			Nds[ty][tx] = 0.0;
		__syncthreads();
		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < TILE_WIDTH; ++k) {
			Pvalue+= Mds[ty][k]*Nds[k][tx];
		}
		__syncthreads();
	}
	if((Row < Width) && (Col < Width))
		d_P[Row*Width+Col] = Pvalue;
}

// Prints out cuda error if error occurs
void checkError(cudaError_t err){
	if(err != cudaSuccess){
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__,__LINE__);
		exit(EXIT_FAILURE);
	}
}

void MatrixMult(float *M, float* N, float* P, int Width) {
	float *d_M, *d_N, *d_P;
	int size = Width * Width * sizeof(float);
	
	struct dim3 blocks((int) ceil(Width/(float) TILE_WIDTH),(int) ceil(Width/(float) TILE_WIDTH),1);
	struct dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH,1);

	cudaMalloc((void**) &d_M, size);
	cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_N, size);
	cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);
	cudaError_t err;
	err = cudaMalloc((void**) &d_P, size);
	checkError(err);
	MatrixMulKernel<<< blocks, threadsPerBlock>>> (d_M, d_N, d_P, Width);
	cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);
	cudaFree(d_M);cudaFree(d_N);cudaFree(d_P);
}

int RandomNum() {
	return rand()%20;
}

void GenMatrices(float *M, float *N, int Width) {
	for(int i = 0; i < Width; ++i) {
		for(int j = 0; j < Width; ++j) {
			M[Width*j+i] = RandomNum();
			N[Width*j+i] = RandomNum();
		}	
	}
}

void MulMatrices(float *M, float *N, float *P, int Width) {
	for(int i = 0; i < Width; ++i) { //row
		for(int j = 0; j < Width; ++j) { //column
				float Pvalue= 0;
				for (int k = 0; k < Width; ++k) {
					Pvalue+= M[i*Width+k]*N[k*Width+j];
				}
				P[i*Width+j] = Pvalue;
		}
	}
}

bool compareMatrices(float *M, float *N, int Width) {
	for(int i = 0; i < Width; ++i) {
		for(int j = 0; j < Width; ++j) {
			if(M[Width*j+i] != N[Width*j+i]) return 0;
		}	
	}
	return 1;
}

int main() {
	srand(100);
	int numOfTest = 1;
	int test[1] = {8500};

	for(int a =0; a < numOfTest; ++a){
		int width = test[a];
		float *m = (float*) malloc(sizeof(float)*width*width);
		float *n = (float*) malloc(sizeof(float)*width*width);
		float *p = (float*) malloc(sizeof(float)*width*width);
		float *p2 = (float*) malloc(sizeof(float)*width*width);
		GenMatrices(m,n, width);
		//printf("Baseline.................\n");
		//MulMatrices(m,n,p2,width);
		//printM(p2,width);
		printf("\n========== test # %d ======\n",a+1);
	printf("Tile width is: %d\n", TILE_WIDTH);
		printf("Tiles per row: %f\n", ceil(width/(float)TILE_WIDTH));
		printf("Matrix width =  %d\n",test[a]);
		MatrixMult(m,n,p, width);
		if(width < 20){
			printM(p2,width);
			printf("Cuda .................\n");
			printM(p, width);
		}
	/*	if(compareMatrices(p,p2,width)) 
			printf("Results are the same!\n");
		else
			printf("Results are not the same...\n"); */

	}
	return 0;
}
