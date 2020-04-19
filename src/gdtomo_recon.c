#include <data/data_types.h>
#include <data/file_io.h>
#include <data/data_ops.h>
#include <reconstructor/reconstructor.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void read_recon_info(const char* recon_info_fn, Data_3d* projs, Data_2d* angles,
  char** recon_fn, char** err_fn, Recon_param* param){

  FILE *recon_info;
  char *line;
  size_t len = 0;
  int ll = 0;

  recon_info = fopen(recon_info_fn, "r");
  if(recon_info == NULL){
    exit(EXIT_FAILURE);
  }
  getline(&line, &len, recon_info);
  
  ll = getline(&line, &len, recon_info);
  if(ll != -1){
    line[ll-1] = '\0';
    import_3d_data(line, projs);
  }
  else{
    exit(EXIT_FAILURE);
  }
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  ll = getline(&line, &len, recon_info);
  if(ll != -1){
    line[ll-1] = '\0';
    import_2d_data(line, angles);
  }
  else{
    exit(EXIT_FAILURE);
  }
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  ll = getline(&line, &len, recon_info);
  if(ll != -1){
    line[ll-1] = '\0';
    *recon_fn = (char *)malloc(ll*sizeof(char));
    strcpy(*recon_fn,line);
  }
  else{
    exit(EXIT_FAILURE);
  }
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  ll = getline(&line, &len, recon_info);
  if(ll != -1){
    line[ll-1] = '\0';
    *err_fn = (char *)malloc(ll*sizeof(char));
    strcpy(*err_fn,line);
  }
  else{
    exit(EXIT_FAILURE);
  }
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  char* ptr;
  if(getline(&line, &len, recon_info)!= -1){
    param->n_iter = strtol(line, &ptr, 10);
  }
  else{
    exit(EXIT_FAILURE);
  }
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  if(getline(&line, &len, recon_info)!= -1){
    param->alpha = strtod(line, &ptr);
  }
  else{
    exit(EXIT_FAILURE);
  }
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  param->recon_dim = malloc(sizeof(unsigned int)*3);

  if(getline(&line, &len, recon_info)!= -1){
    (param->recon_dim)[0] = strtol(line, &ptr, 10);
  }
  else{
    exit(EXIT_FAILURE);
  }
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  if(getline(&line, &len, recon_info)!= -1){
    (param->recon_dim)[1] = strtol(line, &ptr, 10);
  }
  else{
    exit(EXIT_FAILURE);
  }
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  if(getline(&line, &len, recon_info)!= -1){
    (param->recon_dim)[2] = strtol(line, &ptr, 10);
  }
  else{
    exit(EXIT_FAILURE);
  }
  getline(&line, &len, recon_info);
  getline(&line, &len, recon_info);

  if(getline(&line, &len, recon_info)!= -1){
    param->num_cores = strtol(line, &ptr, 10);
  }
  else{
    exit(EXIT_FAILURE);
  }

  fclose(recon_info);
}

int run_gdtomo_recon(const char* recon_info_fn){
  Data_3d projs;
  Data_2d angles;
  char* recon_fn;
  char* err_fn;
  Recon_param param;

  read_recon_info(recon_info_fn, &projs, &angles, &recon_fn, &err_fn, &param);
  printf("%s\n", "Data import complete");

  deg_to_rad(&angles);

  Data_3d recon;
  Data_3d err;
  calc_reconstruction(&recon, &angles, &projs, &err, &param);
  export_3d_data(recon_fn, &recon);
  export_3d_data(err_fn, &err);

  printf("%s\n", "Data export complete");

  free(recon_fn);
  free(err_fn);
  free_3d_data(&projs);
  free_2d_data(&angles);
  free_3d_data(&recon);
  free_3d_data(&err);

  return 0;
}
