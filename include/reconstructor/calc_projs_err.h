#ifndef CALC_PROJS_ERR_H
#define CALC_PROJS_ERR_H

#include <data/data_types.h>

/* Calculates the r1 and r2 projection errors in [proj_diff] with respect to
 * [projs]. */
void calc_projs_err(Data_2d* err, Data_3d* projs_diff, Data_3d* projs);

#endif
