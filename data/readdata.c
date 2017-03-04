#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Reads floating point data into an array from a file where the first 2 lines 
// of the file are the dimensions of the array and remaining lines are the 
// floting point values that fill the m x n arraiy.
// *** REMEMBER TO FREE THE RETURNED ARRA *** //
float * readfile(const char *filename, int *m, int *n) {
  FILE *file = fopen(filename, "r");

  //Read m and n from the first two lines 
  fscanf(file, "%d", m);
  fscanf(file, "%d", n);

  //Allocate the Array
  float *data = (float *) malloc(sizeof(float) * *m * *n);

  //Fill the data
  for(int i = 0; i < *m; i++) {
    for(int j = 0; j < *n; j++) {
      fscanf(file, "%f", &data[i * *n + j]);
    }
  }

  fclose(file);
  return data;
}

/*
//For testing
int main(int argc, char* argv[]) {
  if(argc != 2) {
    printf("Filename Required");
    exit(1);
  }
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
}
*/
