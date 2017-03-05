#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  FILE *file;i
  //Check args
  if((argc != 3 && argc != 4)
  || !(strcmp(argv[1], "CUDA") || strcmp(argv[1], "PHI"))) {
    printf("Usage: ./main [CUDA | PHI] [FILE] [OPTIONAL FILE]\n");
    exit(1);
    }

  //Check that files are valid
  if(!(file = fopen(argv[2], "r")) && (fclose(file))
  && (argc == 4 && !(file = fopen(argv[3], "r")) && fclose(file)))
    printf("Invalid file name");

  //run the program
  if(strcmp(argv[1],"CUDA")) {
    if(argc = 3)
      1 == 1; /* Run CUDA PROGRAM with m = argv[2], n = argv[2] */
    else
      1 == 1; /* Run CUDA PROGRAM with m = argv[2], n = argv[3] */
  } else if(strcmp(argv[1], "PHI")) {
    if(argc = 3)
      1 == 1; /* Run PHI PROGRAM with m = argv[2], n = argv[3] */
    else
      1 == 1; /* Run PHI PROGRAM with m = argv[2], n = argv[3] */
  }
}

