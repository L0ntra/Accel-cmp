#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define NNN 2048
#define MMIC mic:2

__declspec (target(MMIC)) float data_A[NNN][NNN] __attribute__((aligned(64)));
__declspec (target(MMIC)) float data_B[NNN][NNN]__attribute__((aligned(64)));
__declspec (target(MMIC)) float sol[NNN][NNN] __attribute__((aligned(64)));


////VECTOR MATRIX MULTIPLY
void vecMatrixMult() {
  //Alloc
  int i, j, k, numthreads;
  struct timespec start, stop, elap;
  srand(0);

  printf("Initializing\r\n");
  #pragma offload target(mic:2)
  #pragma omp parallel
  #pragma omp master
  numthreads = omp_get_num_threads();

  //Fill Data
  #pragma offload target(MMIC) 
  #pragma omp parallel for private(i, j) shared(data_A, data_B, sol)
  for(i = 0; i < NNN ; i++) 
    for(j = 0; j < NNN; j++) {
      data_A[i][j] = rand();
      data_B[i][j] = rand();
      sol[i][j] = 0;
    }

  printf("Starting Compute on %d threads\r\n",numthreads);
  
  //Calc
  clock_gettime(CLOCK_REALTIME, &start);
  #pragma offload target(MMIC)
  #pragma omp parallel for private(i, j)  
  for(i = 0; i < NNN; i++)
    for(j = 0; j < NNN; j++) {
      #pragma omp simd
      for(k = 0; k < NNN; k++)
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
         NNN, NNN, elap.tv_sec, elap.tv_nsec / 1000000);
}


////MAIN
int main(int argc, char *argv[]) {
//  int n = atoi(argv[1]);
  vecMatrixMult();
}

