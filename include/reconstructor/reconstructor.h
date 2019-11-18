#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H

#include <data/data_types.h>

typedef struct Recon_param{
  double alpha;
  unsigned int num_cores;
  unsigned int n_iter;
  unsigned int *recon_dim;
} Recon_param;

/* Main reconstruction loop using gradient descent.
Final reconstruction is allocated and set in [vol]
With [proj]ections of the reconstructions created with tilt [angles].
[param]eters are defined in the struct above */
void calc_reconstruction(Data_3d* vol, Data_2d* angles, Data_3d* projs,
  Data_3d* err, Recon_param* param);

#endif
