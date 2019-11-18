#include <data/data_types.h>
#include <data/file_io.h>
#include <data/data_ops.h>
#include <reconstructor/proj_calc.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int run_test()
{ 
  Data_2d angles;
  char* angles_filename = "./data/c++/multiple_tilt/angles_mt";
  import_2d_data(angles_filename, &angles);
  deg_to_rad(&angles);

  Data_3d recon;
  char* recon_filename = "./data/c++/recon/recon_true";
  import_3d_data(recon_filename, &recon);

  Data_2d proj;
  unsigned int *proj_dim = (unsigned int *)malloc(2*sizeof(unsigned int));
  proj_dim[0] = 50;
  proj_dim[1] = 50;
  alloc_2d_data(&proj, proj_dim);

  pthread_mutex_t lock;
  if(pthread_mutex_init(&lock,NULL)){
    fprintf(stderr, "Error initializing mutex\n");
    exit(1);
  }
  pthread_mutex_destroy(&lock);
  calc_projection(&recon, (angles.data)[0], &proj, &lock);

  export_2d_data("./data/matlab/cout/proj", &proj);

  free_2d_data(&angles);
  free_3d_data(&recon);
  free_2d_data(&proj);

  return 0;
}
