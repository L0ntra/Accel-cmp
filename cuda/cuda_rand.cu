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
#include "cuda.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void getInfo(int *threadsPerBlock, size_t *sharedMemPerBlock) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  *threadsPerBlock = deviceProp.maxThreadsPerBlock;
  *sharedMemPerBlock = deviceProp.sharedMemPerBlock;
}

__global__
void matrixMultiplyTileKernel(float * A, float * B, float * C, int w) {
  float temp = 0;
  __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
  
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  for(int i = 0; i < (w - 1)/TILE_WIDTH + 1; i++) {
    s_A[ty][tx] = 
      (row < w && i * TILE_WIDTH + tx < w) ? A[row * w + i * TILE_WIDTH + tx] : 0;
    s_B[ty][tx] = 
      (col < w && (i * TILE_WIDTH + ty) < w) ? B[(i * TILE_WIDTH + ty) * w + col] : 0;
    __syncthreads();

    for(int i = 0; i < TILE_WIDTH; i++)
      temp += s_A[ty][i] * s_B[i][tx];
    __syncthreads();
  }

  C[row * w + col] = temp;
}

void matrixMultiplyCUDA(float * A, float * B, float *C, int n, 
                        int threadPerBlock, size_t sharedMemPerBlock) {
  int size = n * n * sizeof(float);
  float *d_A, *d_B, *d_C;
  
  //Allocate
  cudaMalloc((void**) &d_A, size);
  cudaMalloc((void**) &d_B, size);
  cudaMalloc((float**) &d_C, size);

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

  struct timespec start, stop, elap;

  //Read File(s)
  int m, n;
  m = strtol(argv[1],NULL,10);
  n = strtol(argv[2],NULL,10);
  
  float *h_A = (float *) malloc(sizeof(float) * m * n);
  float *h_B = (float *) malloc(sizeof(float) * m * n);
  float *h_C = (float *) malloc(sizeof(float) * m * n);
  
  //Fill the data
  srand(0);
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      h_A[i * m + j] = rand();
      h_B[j * n + i] = rand();
    }
  }

  printf("start\n");
  //Do Computation
  clock_gettime(CLOCK_REALTIME, &start);
  matrixMultiplyCUDA(h_A, h_B, h_C, n, threadPerBlock, sharedMemPerBlock);
  clock_gettime(CLOCK_REALTIME, &stop);

  if((stop.tv_nsec - start.tv_nsec) < 0) {
    elap.tv_sec  = stop.tv_sec  - start.tv_sec  - 1;
    elap.tv_nsec = stop.tv_nsec - start.tv_nsec + 1000000000;
  } else {
    elap.tv_sec  = stop.tv_sec  - start.tv_sec;
    elap.tv_nsec = stop.tv_nsec - start.tv_nsec;
  }

  printf("Time to multoply %dX%d Matrix: %lus %lu microseconds.\n",
    m, n, elap.tv_sec, elap.tv_nsec / 1000000);
 
/*  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++)
      if(h_C[i * m + j] != m)
        printf("%f ", h_C[i * m + j]);
    printf("\n");
  }
*/

  free(h_A); free(h_B); free(h_C);

 return 0;
}


