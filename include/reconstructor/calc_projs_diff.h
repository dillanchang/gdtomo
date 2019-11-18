#ifndef CALC_PROJS_DIFF_H
#define CALC_PROJS_DIFF_H

#include <data/data_types.h>

/* Calculates the projection differences between the projections in [vol] and
 * [projs], and saves it as [projs_diff]. */
void calc_projs_diff(Data_3d* projs_diff, Data_3d* vol, Data_3d* projs, Data_2d*
  angles, unsigned int num_cores);

#endif
