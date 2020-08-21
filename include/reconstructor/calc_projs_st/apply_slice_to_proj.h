#ifndef APPLY_SLICE_TO_PROJ_H
#define APPLY_SLICE_TO_PROJ_H

#include <data/data_types.h>

void apply_slice_to_proj(float* dev_slice, float* dev_proj_arr,
  int* dev_job_wxz, float* dev_job_ww1, unsigned int n_pid, unsigned int y_len,
  unsigned int vdy_slice, unsigned int vdxz, unsigned int pdx, unsigned int Nxz);
  
#endif
