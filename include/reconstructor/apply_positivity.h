#ifndef APPLY_POSITIVITY_H
#define APPLY_POSITIVITY_H

#include <data/data_types.h>

/* 
 * Applies positivity constraint to [recon].
 */
void apply_positivity(Data_3d* recon);

void apply_positivity_st(Data_2d* recon);

#endif
