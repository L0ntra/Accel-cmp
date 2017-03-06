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


#include "../data/readdata.h"
#include <cuda.h>

void getInfo(int *threadsPerBlock, size_t *sharedMemPerBlock) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  *threadsPerBlock = deviceProp.maxThreadsPerBlock;
  *sharedMemPerBlock = deviceProp.sharedMemPerBlock;
}

void matrixMultiply(float A[], float B[], float C[], int m) {
  float temp;
  for(int i = 0; i < m; i++)
    for(int j = 0; j < m; j++) {
      temp = 0;
      for(int k = 0; k < m; k++)
        temp += A[i * m + k] * B[k * m + j];
      C[i * m + j] = temp;
    }
}


__global__ 
void matrixMultiplyKernel(float A[], float B[], float C[], int m) {
  int j = blockDim.x * blockIdx.x + threadIdx.x; //COL
  int i = blockDim.y * blockIdx.y + threadIdx.y; //ROW

  if((i < m) && (j < m)) {
    float temp = 0;
    for(int k = 0; k < m; k++)
      temp += A[i * m + k] * B[k * m + j];
    C[i * m + j] = temp;
  }
}


__global__
void matrixMultiplyTileKernel(float A[], float B[], float C[], int m, unsigned int maxArrSize) {
  int j = blockDim.x * blockIdx.x + threadIdx.x; //COL
  int i = blockDim.y * blockIdx.y + threadIdx.y; //ROW

  float temp = 0;
  extern __shared__ float s_A[];
  extern __shared__ float s_B[];

  if((i < m) && (j < m)) {
    //Copy Data to shared memory
    s_A[i * m + j] = A[i * m + j];
    s_B[j * m + i + maxArrSize] = B[j * m + i];
    __syncthreads();
  
    for(int k = 0; k < m; k++)
      temp += s_A[i * m + k] * s_B[k * m + j + maxArrSize];
    
    C[i * m + j] = temp;
  }
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

  unsigned int maxArrSize = 32;//sharedMemPerBlock / 4;
  float blk = 32.0;
  dim3 dimGrid(ceil(n/blk), ceil(n/blk), 1);
  dim3 dimBlock(blk, blk, 1);

  //PerformCalculation
//  matrixMultiplyKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
  matrixMultiplyTileKernel<<<dimGrid, dimBlock, maxArrSize>>>(d_A, d_B, d_C, n, maxArrSize);

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
  printf("%i, %lu\n", threadPerBlock, sharedMemPerBlock);

  //Read File(s)
  int m, n;
  char *filename = argv[1];
  float *h_A = readfile(filename, &m, &n);
  float *h_C = (float *) malloc(sizeof(float) * m * n);
  float *h_D = (float *) malloc(sizeof(float) * m * n);

  //Do Computation
  matrixMultiplyCUDA(h_A, h_A, h_C, n, threadPerBlock, sharedMemPerBlock);
  matrixMultiply(h_A, h_A, h_D, n);

  //Print Solution
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      printf("%f = %f (%s) ", 
        h_C[i * m + j], 
        h_D[i * m + j],
        (h_C[i * m + j] == h_D[i * m + j])? "True" : "False");
    }
    printf("\n");
  }

  free(h_A); free(h_C); free(h_D);

 return 0;
}

