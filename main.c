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

