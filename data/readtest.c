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

