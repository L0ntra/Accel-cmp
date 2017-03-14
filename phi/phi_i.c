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
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "../data/readdata.h"

////VECTOR MATRIX MULTIPLY
void vecMatrixMult(int data_A[], int data_B[], int n) {

  int i, j, k;
  struct timespec start, stop, elap;

  int (* restrict sol)[n] __attribute__((aligned(64)))= malloc(sizeof(float) * n * n);

  #pragma omp parallel for private(i, j) shared(sol, data_A, data_B, n)
  for(i = 0; i < n; i++) 
    for(j = 0; j < n; j++)
      sol[i][j] = 0;
  
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
 
  printf("Time to multoply %dX%d Matrix: %lus %lu microseconds.\n", 
         n, n, elap.tv_sec, elap.tv_nsec / 1000000);
}

////MAIN
int main(int argc, char *argv[]) {
  //Read File
  int m, n;
  char *filename = argv[1];

  int (* restrict data_A)[n] __attribute__((aligned(64))) = 
    readfile_int(filename, &m, &n);
  int (* restrict data_B)[n] __attribute__((aligned(64))) = 
    readfile_int(filename, &m, &n);

  int n = atoi(argv[1]);
  vecMatrixMult(data_A, data_B, n);
}

