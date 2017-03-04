#include "readdata.c"
//For testing purposes
int main(int argc, char* argv[]) {
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

