#include <test/test.h>

#include <stdio.h>
#include <stdlib.h>
#include <data/data_types.h>
#include <data/file_io.h>
#include <data/data_ops.h>
#include <reconstructor/calc_projs_cpu/calc_projs_cpu.h>
#include <reconstructor/calc_projs_gpu/calc_projs_gpu.h>

int run_test()
{
  Data_3d recon;
  import_3d_data("./analysis/20200427_calc_proj_gpu/data/recon.npy", &recon);

  Data_2d angles;
  import_2d_data("./analysis/20200427_calc_proj_gpu/data/angles.npy", &angles);

  Data_3d projs_0;
  unsigned int *projs_0_dim = (unsigned int *)malloc(3*sizeof(unsigned int));
  projs_0_dim[0] = (angles.dim)[0];
  projs_0_dim[1] = (recon.dim)[0];
  projs_0_dim[2] = (recon.dim)[1];
  alloc_3d_data(&projs_0, projs_0_dim);

  Data_3d projs_1;
  unsigned int *projs_1_dim = (unsigned int *)malloc(3*sizeof(unsigned int));
  projs_1_dim[0] = (angles.dim)[0];
  projs_1_dim[1] = (recon.dim)[0];
  projs_1_dim[2] = (recon.dim)[1];
  alloc_3d_data(&projs_1, projs_1_dim);


  printf("%s\n", "starting projection calculation");
  // calc_projs_cpu(&projs_0, &recon, &angles, 6);
  // printf("%s\n", "calc_projs_cpu complete");
  calc_projs_gpu(&projs_1, &recon, &angles);
  printf("%s\n", "calc_projs_gpu complete");

  // export_3d_data("./analysis/20200427_calc_proj_gpu/data/projs_0.npy", &projs_0);
  export_3d_data("./analysis/20200427_calc_proj_gpu/data/projs_1.npy", &projs_1);

  free_3d_data(&projs_0);
  free_3d_data(&projs_1);
  free_3d_data(&recon);
  free_2d_data(&angles);
  return 0;
}

