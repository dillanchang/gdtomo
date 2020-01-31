#ifndef DATA_TYPES_H
#define DATA_TYPES_H

typedef struct Data_2d{
  unsigned int* dim;
  double** data;
} Data_2d;

typedef struct Data_3d{
  unsigned int* dim;
  double*** data;
} Data_3d;

/*
Allocates an empty data set at d.
*/
void alloc_2d_data(Data_2d* d, unsigned int* dim);

void alloc_3d_data(Data_3d* d, unsigned int* dim);

/*
Frees all allocated memory in data types.
*/
void free_2d_data(Data_2d* d);

void free_3d_data(Data_3d* d);

#endif
