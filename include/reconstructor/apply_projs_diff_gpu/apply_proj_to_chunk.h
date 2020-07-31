#ifndef APPLY_PROJ_TO_CHUNK_H
#define APPLY_PROJ_TO_CHUNK_H

#include <data/data_types.h>

/* 
 * 
 */
void apply_proj_to_chunk(double* projs_diff, unsigned int n_proj,
  unsigned int n_proj_tot, unsigned int pdx, unsigned int pdy, double* chunk,
  unsigned int dim_chunk, unsigned int x0, unsigned int y0, unsigned int z0,
  unsigned int vdx, unsigned int vdy, unsigned int vdz, double* r_hats);

#endif
