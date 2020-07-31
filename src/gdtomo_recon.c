#include <data/data_types.h>
#include <data/file_io.h>
#include <data/data_ops.h>
#include <reconstructor/reconstructor.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void read_recon_info(const char* recon_info_fn, Data_3d* projs, Data_2d* angles,
  char** recon_fn, char** err_fn, char** projs_final_fn, Recon_param* param){

  FILE *recon_info;
  char *line;
  size_t len = 0;
  int ll = 0;

  // Opening File
  recon_info = fopen(recon_info_fn, "r");
  if(recon_info == NULL){
    exit(EXIT_FAILURE);
  }
  getline(&line, &len, recon_info);
  
  // Projection path
  ll = getline(&line, &len, recon_info);
  line[ll-1] = '\0';
  import_3d_data(line, projs);
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  // Angles path
  ll = getline(&line, &len, recon_info);
  line[ll-1] = '\0';
  import_2d_data(line, angles);
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  // Recon path
  ll = getline(&line, &len, recon_info);
  line[ll-1] = '\0';
  *recon_fn = (char *)malloc(ll*sizeof(char));
  strcpy(*recon_fn,line);
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  // Err path
  ll = getline(&line, &len, recon_info);
  line[ll-1] = '\0';
  *err_fn = (char *)malloc(ll*sizeof(char));
  strcpy(*err_fn,line);
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  // Final projections path
  ll = getline(&line, &len, recon_info);
  line[ll-1] = '\0';
  *projs_final_fn = (char *)malloc(ll*sizeof(char));
  strcpy(*projs_final_fn,line);
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  char* ptr;
  // N_iter
  getline(&line, &len, recon_info);
  param->n_iter = strtol(line, &ptr, 10);
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  // Alpha
  getline(&line, &len, recon_info);
  param->alpha = strtod(line, &ptr);
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  param->recon_dim = malloc(sizeof(unsigned int)*3);

  // Dim x
  getline(&line, &len, recon_info);
  (param->recon_dim)[0] = strtol(line, &ptr, 10);
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  // Dim y
  getline(&line, &len, recon_info);
  (param->recon_dim)[1] = strtol(line, &ptr, 10);
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  // Dim z
  getline(&line, &len, recon_info);
  (param->recon_dim)[2] = strtol(line, &ptr, 10);
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  // Close file
  fclose(recon_info);
}

int run_gdtomo_recon(const char* recon_info_fn){
  Data_3d projs;
  Data_2d angles;
  char* recon_fn;
  char* err_fn;
  char* projs_final_fn;
  Recon_param param;

  read_recon_info(recon_info_fn, &projs, &angles, &recon_fn, &err_fn, &projs_final_fn, &param);
  printf("%s\n", "Data import complete");

  deg_to_rad(&angles);

  Data_3d recon;
  Data_3d err;
  Data_3d projs_final;
  calc_reconstruction(&recon, &angles, &projs, &err, &projs_final, &param);
  export_3d_data(recon_fn, &recon);
  export_3d_data(err_fn, &err);
  export_3d_data(projs_final_fn, &projs_final);

  printf("%s\n", "Data export complete");

  free(recon_fn);
  free(err_fn);
  free(projs_final_fn);
  free_3d_data(&projs);
  free_2d_data(&angles);
  free_3d_data(&recon);
  free_3d_data(&err);
  free_3d_data(&projs_final);

  return 0;
}

