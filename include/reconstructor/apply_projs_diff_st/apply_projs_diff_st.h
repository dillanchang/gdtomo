#ifndef APPLY_PROJS_DIFF_ST_H
#define APPLY_PROJS_DIFF_ST_H

#include <data/data_types.h>

void apply_projs_diff_st(Data_2d *recon, Data_3d *projs_diff, Data_2d *angles,
  int* wx, float* ww2, float alpha, unsigned int vdz);

#endif
