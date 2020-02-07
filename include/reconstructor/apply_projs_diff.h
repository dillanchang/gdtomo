#ifndef APPLY_PROJS_DIFF_H
#define APPLY_PROJS_DIFF_H

#include <data/data_types.h>

/* 
 * Applies the projection differences, [projs_diff], to [vol]. The projecion
 * differences are aligned by Euler [angles], and the update scalar is defined
 * by [alpha]. This step is the last step in an iteration of gdtomo.
 */
void apply_projs_diff(Data_3d* vol, Data_3d* projs_diff, Data_2d* angles, 
  double alpha, unsigned int num_cores);

#endif
