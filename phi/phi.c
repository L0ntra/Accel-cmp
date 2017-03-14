#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>
//#include "mkl.h"

////VECTOR MATRIX MULTIPLY
void vecMatrixMult(int n) {
  //Alloc
  int i, j, k;
  struct timespec start, stop, elap;

  float (* /*restrict */ data_A)[n] __attribute__((aligned(64))) = malloc(sizeof(float) * n * n);
  float (* /*restrict */ data_B)[n] __attribute__((aligned(64))) = malloc(sizeof(float) * n * n);
  float (*restrict sol)[n] __attribute__((aligned(64)))= malloc(sizeof(float) * n * n);

  //Fill Data
  #pragma omp parallel for private(i, j) shared(sol, data_A, data_B, n)
  for(i = 0; i < n; i++) 
    for(j = 0; j < n; j++) {
      data_A[i][j] = i * n + j + 1;
      data_B[i][j] = j * n + i + 1;
      sol[i][j] = 0;
    }
  
  //Calc
  clock_gettime(CLOCK_REALTIME, &start);
  #pragma omp parallel for private(i, j, k) shared(sol, data_A, data_B, n)
  for(i = 0; i < n; i++)
    for(j = 0; j < n; j++) {
      sol[i][j] = 0;
      #pragma omp simd
      for(k = 0; k < n; k++)
        sol[i][j] += data_A[i][k] * data_B[k][j]; 
    }
  clock_gettime(CLOCK_REALTIME, &stop);
  
  if((stop.tv_nsec - start.tv_nsec) < 0) {
    elap.tv_sec  = stop.tv_sec  - start.tv_sec  - 1;
    elap.tv_nsec = stop.tv_nsec - start.tv_nsec + 1000000000;
  } else {
    elap.tv_sec  = stop.tv_sec  - start.tv_sec;
    elap.tv_nsec = stop.tv_nsec - start.tv_nsec;
  }
 
  //print
  printf("Time to multoply %dX%d Matrix: %lus %lu microseconds.\n", 
         n, n, elap.tv_sec, elap.tv_nsec / 1000000);
  /*
  for(int i = 0; i < n; i++)
    for(int j = 0; j <n; j++) {
      printf("%f\n", sol[i][j]);
    }
  */
}


////MAIN
int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);
  vecMatrixMult(n);
}

