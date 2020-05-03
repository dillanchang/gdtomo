#ifndef CALC_PROJS_1_H
#define CALC_PROJS_1_H

#include <data/data_types.h>

/* 
 * Calculates [proj]ection[s] of [vol]ume from viewing from orientation defined
 * by Euler [angles].
 */
void calc_projs_1(Data_3d* projs, Data_3d* vol, Data_2d* angles);

#endif
