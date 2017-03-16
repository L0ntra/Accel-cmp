/*
  Copyright (c) 2017 
  Gregory Gaston, Geoffrey Maggi, Prajyoth Bhandary, and Sriharsha Makineni

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
extern "C" { //Fixes linking issue with c files
	#include "../data/readdata.h"
}
#include "cuda.h"
#include <cuda.h>
#include <sys/time.h>

void getInfo(int *threadsPerBlock, size_t *sharedMemPerBlock) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  *threadsPerBlock = deviceProp.maxThreadsPerBlock;
  *sharedMemPerBlock = deviceProp.sharedMemPerBlock;
}

__global__
void matrixMultiplyTileKernel(float A[], float B[], float C[], int w) {
  int j = blockDim.x * blockIdx.x + threadIdx.x; //COL
  int i = blockDim.y * blockIdx.y + threadIdx.y; //ROW

  float temp = 0;
  __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];


  for(int k = 0; k < w/TILE_WIDTH; k++) {
    s_A[threadIdx.y][threadIdx.x] = A[i * w + k * TILE_WIDTH + threadIdx.x];
    s_B[threadIdx.y][threadIdx.x] = B[(k * TILE_WIDTH + threadIdx.y) * w + j];
    __syncthreads();

    for(int h = 0; h < TILE_WIDTH; h++)
      temp += s_A[threadIdx.y][h] * s_B[h][threadIdx.x];
    __syncthreads();
  }

  C[i * w + j] = temp;
}

void matrixMultiplyCUDA(float A[], float B[], float C[], int n, 
                        int threadPerBlock, size_t sharedMemPerBlock) {
  int size = n * n * sizeof(float);
  float *d_A, *d_B, *d_C;
  
  //Allocate
  cudaMalloc((void**) &d_A, size);
  cudaMalloc((void**) &d_B, size);
  cudaMalloc((void**) &d_C, size);

  //Copy Memory
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  float blk = 32.0;
  dim3 dimGrid(ceil(n/blk), ceil(n/blk), 1);
  dim3 dimBlock(blk, blk, 1);

  //PerformCalculation
  matrixMultiplyTileKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

  //Copy Solution
  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  //Free
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

int main(int argc, char *argv[]) {
  //Read Device
  int threadPerBlock;
  size_t sharedMemPerBlock;
  getInfo(&threadPerBlock, &sharedMemPerBlock);

  timeval start, stop;

  //Read File(s)
  int m, n;
  char *filename = argv[1];
  float *h_A = readfile(filename, &m, &n);
  float *h_B = readfile_transpose(filename, &m, &n);
  float *h_C = (float *) malloc(sizeof(float) * m * n);

  //Do Computation
  gettimeofday(&start, NULL);
  matrixMultiplyCUDA(h_A, h_B, h_C, n, threadPerBlock, sharedMemPerBlock);
  gettimeofday(&stop, NULL);

  printf("Time to run: %lu microseconds\n", stop.tv_usec - start.tv_usec);

  free(h_A); free(h_B); free(h_C);

 return 0;
}

