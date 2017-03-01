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

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

void dispInfo() {
  int deviceCount;
  cudaDeviceProp deviceProp;
  cudaGetDeviceCount(&deviceCount);
  printf("Num Cuda Devices: %i\n",deviceCount);
  for(int i = 0; i <  deviceCount; i++) {
    cudaGetDeviceProperties(&deviceProp, i);
    printf("Name:         %s\n", deviceProp.name);
    printf("Thrads/Block: %i\n", deviceProp.maxThreadsPerBlock);
    printf("M.m:	  %i.%i\n", deviceProp.major, deviceProp.minor);
  }
}


void matrixMultiply(int* A, int* B, int* C, int n) {
  for(int i = 0; i < n; i++)
    for(int j = 0; j < n; j++)
      for(int k = 0; k < n; k++)
        C[i * n + j] += A[i * n + k] * B[k * n + j];
}

__global__ 
void matrixMultiplyKernel(int* A, int* B, int* C, int n) {
  int j = blockDim.x * blockIdx.x + threadIdx.x; //ROW
  int i = blockDim.y * blockIdx.y + threadIdx.y; //COL

  if((i < n) && (j < n)) {
    int temp = 0;
    for(int k = 0; k < n; k++)
      temp += A[i * n + k] * B[k * n + j];
    C[i * n + j] = temp;
  }
}

void matrixMultiplyCUDA(int* A, int* B, int* C, int n) {
  int size = n * n * sizeof(int);
  int *d_A, *d_B, *d_C;
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
  dispInfo();
  int n;
  if(argc != 2
  || (n = (int) strtol(argv[1], (char**) NULL, 10)) == 1
  || n < 1) {
    printf("Error: Call with int > 0\n");
    return 1;
  }

  int h_A[n][n], h_B[n][n], h_C[n][n], h_D[n][n];
  for(int i = 0; i < n; i++)
    for(int j = 0; j < n; j++) {
      h_A[i][j] = h_B[i][j] = i * n + j;
      h_C[i][j] = h_D[i][j] = 0;
    }

  matrixMultiply((int *) &h_A, (int *) &h_B, (int *) &h_C, n);
  matrixMultiplyCUDA((int *) &h_A, (int *) &h_B, (int *) &h_D, n);
  
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++)
      printf("%i ", h_A[i][j]);
    printf("\n");
  }
  printf("\n");

  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++)
      printf("%i ", h_B[i][j]);
    printf("\n");
  }
  printf("\n");

  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++)
      printf("%i ", h_C[i][j]);
    printf("\n");
  }
  printf("\n");

  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++)
      printf("%i ", h_D[i][j]);
    printf("\n");
  }  

  return 0;
}
