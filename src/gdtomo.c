#include <data/data_types.h>
#include <data/file_io.h>
#include <data/data_ops.h>
#include <reconstructor/reconstructor.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <test/test.h>

int main(int argc, char** argv)
{
  if(argc > 1 && (strcmp(argv[1],"test")==0)){
    return run_test();
  }

  printf("%s\n", "#===========================================================================#");
  printf("%s\n", "#                                  gdtomo                                   #");
  printf("%s\n", "#                                                                           #");
  printf("%s\n", "#                                            github.com/dillanchang/gdtomo  #");
  printf("%s\n", "#===========================================================================#");

  Data_2d angles;
  char* angles_filename = "./data/c++/NiPt/angles";
  import_2d_data(angles_filename, &angles);
  deg_to_rad(&angles);

  Data_3d proj;
  char* proj_filename = "./data/c++/NiPt/proj";
  import_3d_data(proj_filename, &proj);

  Data_3d recon;

  Data_3d err;

  Recon_param param;
  param.num_cores  = 12;
  param.alpha      = 0.5;
  param.n_iter     = 100;
  param.recon_dim  = malloc(sizeof(unsigned int)*3);
  (param.recon_dim)[0] = 200;
  (param.recon_dim)[1] = 200;
  (param.recon_dim)[2] = 200;

  printf("%s\n", "Data import complete");

  calc_reconstruction(&recon, &angles, &proj, &err, &param);
  export_3d_data("./data/matlab/cout/recon", &recon);
  export_3d_data("./data/matlab/cout/err", &err);

  printf("%s\n", "Data export complete");

  free_2d_data(&angles);
  free_3d_data(&proj);
  free_3d_data(&recon);
  free_3d_data(&err);

  return 0;
}

