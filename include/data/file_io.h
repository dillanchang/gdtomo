#ifndef FILE_IO_H
#define FILE_IO_H

#include <data/data_types.h>

/*
Allocates and imports a type double data set located at [filename]+".csv",
with dimensions defined at [*dim].
The imported data will be loaded at [*data].
*/
void import_2d_data(char* filename, Data_2d* d);

void import_3d_data(char* filename, Data_3d* d);

/*
Exports a double [data] set to [filename]+".csv",
with metadata written to [filename]+"_meta.txt",
with dimensions defined at [dim].
*/
void export_2d_data(char* filename, Data_2d* d);

void export_3d_data(char* filename, Data_3d* d);

#endif

