#ifndef GET_CHUNK_OVERLAP_H
#define GET_CHUNK_OVERLAP_H

#include <data/data_types.h>

/* 
 * Returns the correct [chunk] given the queried [chunk_idx]. If [chunk_idx] is
 * out of bounds, the method returns 1. Else, returns 0.
 */
int get_chunk_overlap(Data_3d* vol, unsigned int chunk_idx, double* chunk,
  unsigned int dim_chunk, double* chunk_origin);

#endif
