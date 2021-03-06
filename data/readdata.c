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

#include "readdata.h"

// Reads floating point data into an array from a file where the first 2 lines 
// of the file are the dimensions of the array and remaining lines are the 
// floating point values that fill the m x n array.
// *** REMEMBER TO FREE THE RETURNED ARRAY *** //
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
      fscanf(file, "%f", &data[i * (*n) + j]);
    }
  }

  fclose(file);
  return data;
}

// Reads floating point data into a transposed array from a file where the first 
// 2 lines of the file are the dimensions of the array and remaining lines are the 
// floating point values that fill the m x n array.
// *** REMEMBER TO FREE THE RETURNED ARRAY *** //
float * readfile_transpose(const char *filename, int *m, int *n) {
  FILE *file = fopen(filename, "r");

  //Read m and n from the first two lines 
  fscanf(file, "%d", m);
  fscanf(file, "%d", n);

  //Allocate the Array
  float *data = (float *) malloc(sizeof(float) * *m * *n);

  //Fill the data
  for(int i = 0; i < *m; i++) {
    for(int j = 0; j < *n; j++) {
      fscanf(file, "%f", &data[j * (*n) + i]);
    }
  }

  fclose(file);
  return data;
}

// Reads integer data into an array from a file where the first 2 lines 
// of the file are the dimensions of the array and remaining lines are the 
// integer point values that fill the m x n array.
// *** REMEMBER TO FREE THE RETURNED ARRAY *** //
int * readfile_int(const char *filename, int *m, int *n) {
  FILE *file = fopen(filename, "r");

  //Read m and n from the first two lines 
  fscanf(file, "%d", m);
  fscanf(file, "%d", n);

  //Allocate the Array
  int *data = (int *) malloc(sizeof(int) * *m * *n);

  //Fill the data
  for(int i = 0; i < *m; i++) {
    for(int j = 0; j < *n; j++) {
      fscanf(file, "%d", &data[i * (*n) + j]);
    }
  }

  fclose(file);
  return data;
}

// Reads integer data into a transposed array from a file where the first 
// 2 lines of the file are the dimensions of the array and remaining lines are the 
// integer values that fill the m x n array.
// *** REMEMBER TO FREE THE RETURNED ARRAY *** //
int * readfile_int_transpose(const char *filename, int *m, int *n) {
  FILE *file = fopen(filename, "r");

  //Read m and n from the first two lines 
  fscanf(file, "%d", m);
  fscanf(file, "%d", n);

  //Allocate the Array
  int *data = (int *) malloc(sizeof(int) * *m * *n);

  //Fill the data
  for(int i = 0; i < *m; i++) {
    for(int j = 0; j < *n; j++) {
      fscanf(file, "%d", &data[j * (*n) + i]);
    }
  }

  fclose(file);
  return data;
}


// Reads Double precision Floating Point data into an array from a file where the first 
// 2 lines of the file are the dimensions of the array and remaining lines are the Double
// precision floating point values that fill the m x n array.
// *** REMEMBER TO FREE THE RETURNED ARRAY *** //
double * readfile_double(const char *filename, int *m, int *n) {
  FILE *file = fopen(filename, "r");

  //Read m and n from the first two lines 
  fscanf(file, "%d", m);
  fscanf(file, "%d", n);

  //Allocate the Array
  double *data = (double *) malloc(sizeof(double) * *m * *n);

  //Fill the data
  for(int i = 0; i < *m; i++) {
    for(int j = 0; j < *n; j++) {
      fscanf(file, "%lf", &data[i * (*n) + j]);
    }
  }

  fclose(file);
  return data;
}

// Reads Double precision Floating Point data into a transposed array from a file where 
// the first 2 lines are the dimensions of the array and remaining lines are the Double
// precision floating point values that fill the m x n array.
// *** REMEMBER TO FREE THE RETURNED ARRAY *** //
double * readfile_double_transpose(const char *filename, int *m, int *n) {
  FILE *file = fopen(filename, "r");

  //Read m and n from the first two lines 
  fscanf(file, "%d", m);
  fscanf(file, "%d", n);

  //Allocate the Array
  double *data = (double *) malloc(sizeof(double) * *m * *n);

  //Fill the data
  for(int i = 0; i < *m; i++) {
    for(int j = 0; j < *n; j++) {
      fscanf(file, "%lf", &data[j * (*n) + i]);
    }
  }

  fclose(file);
  return data;
}
