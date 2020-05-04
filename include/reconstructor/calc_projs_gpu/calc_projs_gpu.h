#ifndef CALC_PROJS_GPU_H
#define CALC_PROJS_GPU_H

#include <data/data_types.h>

/* 
 * Calculates [proj]ection[s] of [vol]ume from viewing from orientation defined
 * by Euler [angles].
 */
void calc_projs_gpu(Data_3d* projs, Data_3d* vol, Data_2d* angles);

#endif
