#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define NNN 1024

__declspec (target(mic:2)) float data_A[NNN][NNN] __attribute__((aligned(64)));
__declspec (target(mic:2)) float data_B[NNN][NNN]__attribute__((aligned(64)));
__declspec (target(mic:2)) float sol[NNN][NNN] __attribute__((aligned(64)));


////VECTOR MATRIX MULTIPLY
void vecMatrixMult() {
  //Alloc
  int i, j, k;
  struct timespec start, stop, elap;

  //Fill Data
  #pragma offload target(mic:2) 
  #pragma omp parallel for private(i, j) shared(data_A, data_B, sol)
  for(i = 0; i < NNN ; i++) 
    for(j = 0; j < NNN; j++) {
      data_A[i][j] = 1.1;
      data_B[i][j] = 1.1;
      sol[i][j] = 0;
    }
  
  //Calc
  clock_gettime(CLOCK_REALTIME, &start);
  #pragma offload target(mic:2)
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

