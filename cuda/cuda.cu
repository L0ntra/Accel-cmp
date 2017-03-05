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

void getInfo(int *threadsPerBlock) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  *threadsPerBlock = deviceProp.maxThreadsPerBlock;
}

void matrixMultiply(int* A, int* B, int* C, int n) {
  for(int i = 0; i < n; i++)
    for(int j = 0; j < n; j++)
      for(int k = 0; k < n; k++)
        C[i * n + j] += A[i * n + k] * B[k * n + j];
}

__global__ 
void matrixMultiplyKernel(float* A, float* B, float* C, int n) {
  int j = blockDim.x * blockIdx.x + threadIdx.x; //ROW
  int i = blockDim.y * blockIdx.y + threadIdx.y; //COL

  if((i < n) && (j < n)) {
    int temp = 0;
    for(int k = 0; k < n; k++)
      temp += A[i * n + k] * B[k * n + j];
    C[i * n + j] = temp;
  }
}

void matrixMultiplyCUDA(float *A, float *B, float *C, int n) {
  int size = n * n * sizeof(float);
  float *d_A, *d_B, *d_C;
  
  //Allocate
  cudaMalloc((void**) &d_A, size);
  cudaMalloc((void**) &d_B, size);
  cudaMalloc((void**) &d_C, size);

  //Copy Memory
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil(n/32.0), ceil(n/32.0), 1);
  dim3 dimBlock(32, 32, 1);

  //PerformCalculation
  matrixMultiplyKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

  //Copy Solution
  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  //Free
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

int main(int argc, char *argv[]) {
//Read Device
  int blockPerThread;
  getInfo(&blockPerThread);
  printf("%i\n", blockPerThread);

//Read File
  int m, n;
  char *filename = argv[1];
  float *data = readfile(filename, &n, &m);

  printf("%i, %i, \n", m, n);
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j ++) {
      printf("%f ", data[i * n + j]);
      }
    printf("\n");
  }

  free(data);

  /*
  matrixMultiply((float *) &h_A, (float *) &h_B, (float *) &h_C, n);
  */


 return 0;
}

