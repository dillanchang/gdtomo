#ifndef APPLY_CHUNK_TO_PROJ_H
#define APPLY_CHUNK_TO_PROJ_H

#include <data/data_types.h>

/* 
 * 
 */
void apply_chunk_to_proj(double* dev_chunk, unsigned int dim_chunk,
  double* dev_chunk_origin, double* dev_projs, double* dev_r_hats,
  unsigned int num_projs, unsigned int dim_proj_x, unsigned int dim_proj_y,
  unsigned int lim_proj_z);

#endif
