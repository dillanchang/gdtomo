#include <stdio.h>
#include <data/file_io.h>

int run_test()
{ 
  char* recon_filename      = "./analysis/20200426_Pd-proj/data/recon_dennis.npy";
  char* angles_filename     = "./analysis/20200426_Pd-proj/data/angles.npy";
  char* projs_in_filename   = "./analysis/20200426_Pd-proj/data/projs.npy";

  Data_3d recon;
  import_3d_data(recon_filename, &recon);

  Data_2d angles;
  import_2d_data(angles_filename, &angles);

  Data_3d projs_in;
  import_3d_data(projs_in_filename, &projs_in);

  free_3d_data(&recon);
  free_2d_data(&angles);
  free_3d_data(&projs_in);

  return 0;
}

