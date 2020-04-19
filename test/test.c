#include <stdio.h>
#include <data/file_io.h>

int run_test()
{ 
  char* angles_filename     = "./data/python/multiple_tilt/angles.npy";
  char* proj_filename       = "./data/python/multiple_tilt/projs.npy";
  char* angles_filename_new = "./data/python/multiple_tilt/angles_new.npy";
  char* proj_filename_new   = "./data/python/multiple_tilt/projs_new.npy";

  Data_2d angles;
  import_2d_data(angles_filename, &angles);

  Data_3d projs;
  import_3d_data(proj_filename, &projs);

  export_2d_data(angles_filename_new, &angles);
  export_3d_data(proj_filename_new, &projs);

  free_2d_data(&angles);
  free_3d_data(&projs);

  return 0;
}
