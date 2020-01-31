#include <data/data_types.h>

#include <stdlib.h>

void alloc_2d_data(Data_2d* d, unsigned int* dim){
  d->dim  = dim;
  d->data = (double**)malloc(dim[0]*sizeof(double*));
  for(unsigned int i=0; i<dim[0]; i++){
    (d->data)[i] = (double*)malloc(dim[1]*sizeof(double));
  }
}

void alloc_3d_data(Data_3d* d, unsigned int* dim){
  d->dim  = dim;
  d->data = (double***)malloc(dim[0]*sizeof(double**));
  for(unsigned int i=0; i<dim[0]; i++){
    (d->data)[i] = (double**)malloc(dim[1]*sizeof(double*));
    for(unsigned int j=0; j<dim[1]; j++){
      (d->data)[i][j] = (double*)malloc(dim[2]*sizeof(double));
    }
  }
}

void free_2d_data(Data_2d* d){
  for(unsigned int i=0; i<(d->dim)[0]; i++){
    free((d->data)[i]);
  }
  free(d->data);
  free(d->dim);
}

void free_3d_data(Data_3d* d){
  for(unsigned int i=0; i<(d->dim)[0]; i++){
    for(unsigned int j=0; j<(d->dim)[1]; j++){
      free((d->data)[i][j]);
    }
    free((d->data)[i]);
  }
  free(d->data);
  free(d->dim);
}

