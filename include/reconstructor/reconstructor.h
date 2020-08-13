#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H

#include <data/data_types.h>

typedef struct Recon_param{
  unsigned int n_iter;
  unsigned int *recon_dim;
  double alpha;
} Recon_param;

/* 
 * Main reconstruction loop using gradient descent. Final reconstruction is
 * allocated and set in [vol] With [proj]ections of the reconstructions created
 * with Euler [angles]. [param]eters are defined in the struct above.
 */
void calc_reconstruction(Data_3d* vol, Data_2d* angles, Data_3d* projs,
  Data_3d* err, Data_3d* projs_final, Recon_param* param);

#endif
