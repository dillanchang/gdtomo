#ifndef FILE_IO_H
#define FILE_IO_H

#include <data/data_types.h>

/*
Allocates and imports an npy data set located at [filename]
*/
void import_2d_data(char* filename, Data_2d* d);

void import_3d_data(char* filename, Data_3d* d);

/*
Exports a dataset [d] to [filename] with npy file specifications
*/
void export_2d_data(char* filename, Data_2d* d);

void export_3d_data(char* filename, Data_3d* d);

#endif

