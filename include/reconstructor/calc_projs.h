#ifndef PROJ_CALC_H
#define PROJ_CALC_H

#include <data/data_types.h>

/*
Calculates [proj]ection[s] of [vol]ume from viewing from orientation
defined by euler [angles].
*/
void calc_projections(Data_3d* vol, Data_2d* angles, Data_3d* projs);

#endif
