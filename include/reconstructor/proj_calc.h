#ifndef PROJ_CALC_H
#define PROJ_CALC_H

#include <data/data_types.h>

/*
Calculates [proj]ections of [vol]ume with [dim]ension from viewing at from
orientation defined by euler [angles]
*/
void calc_projection(Data_3d* vol, double* angles, Data_2d* proj);

#endif
