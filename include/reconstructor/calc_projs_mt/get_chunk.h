#ifndef GET_CHUNK_H
#define GET_CHUNK_H

#include <data/data_types.h>

/* 
 * Returns the correct [chunk] given the queried [chunk_idx]. If [chunk_idx] is
 * out of bounds, the method returns 1. Else, returns 0.
 */
int get_chunk(Data_3d* vol, unsigned int chunk_idx, float* chunk,
  unsigned int dim_chunk, float* chunk_origin);

#endif
