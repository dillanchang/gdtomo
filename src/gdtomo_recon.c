#include <data/data_types.h>
#include <data/file_io.h>
#include <data/data_ops.h>
#include <reconstructor/reconstructor.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int run_gdtomo_recon(const char* recon_info_fn){
  FILE *recon_info;
  char *line;
  size_t len = 0;
  unsigned int line_len = 0;
  char *recon_fn;
  char *err_fn;

  recon_info = fopen(recon_info_fn, "r");
  if(recon_info == NULL)
    exit(EXIT_FAILURE);
  
  Data_3d projs;
  line_len = 0;
  if(getline(&line, &len, recon_info) != -1){
    while(line[line_len] != 0){
      line_len = line_len + 1;
    }
    line[line_len-1] = '\0';
    import_3d_data(line, &projs);
  }
  else{
    exit(EXIT_FAILURE);
  }

  Data_2d angles;
  line_len = 0;
  if(getline(&line, &len, recon_info) != -1){
    while(line[line_len] != 0){
      line_len = line_len + 1;
    }
    line[line_len-1] = '\0';
    import_2d_data(line, &angles);
  }
  else{
    exit(EXIT_FAILURE);
  }
  deg_to_rad(&angles);

  Data_2d param_data;
  line_len = 0;
  if(getline(&line, &len, recon_info) != -1){
    while(line[line_len] != 0){
      line_len = line_len + 1;
    }
    line[line_len-1] = '\0';
    import_2d_data(line, &param_data);
  }
  else{
    exit(EXIT_FAILURE);
  }
  Recon_param param;
  param.n_iter         = (unsigned int)(param_data.data)[0][0];
  param.alpha          = (param_data.data)[0][1];
  param.recon_dim      = malloc(sizeof(unsigned int)*3);
  (param.recon_dim)[0] = (unsigned int)(param_data.data)[0][2];
  (param.recon_dim)[1] = (unsigned int)(param_data.data)[0][3];
  (param.recon_dim)[2] = (unsigned int)(param_data.data)[0][4];
  param.num_cores      = (unsigned int)(param_data.data)[0][5];
  free_2d_data(&param_data);

  line_len = 0;
  if(getline(&line, &len, recon_info) != -1){
    while(line[line_len] != 0){
      line_len = line_len + 1;
    }
    line[line_len-1] = '\0';
    recon_fn = (char *)malloc(line_len*sizeof(char));
    strcpy(recon_fn,line);
  }
  else{
    exit(EXIT_FAILURE);
  }

  line_len = 0;
  if(getline(&line, &len, recon_info) != -1){
    while(line[line_len] != 0){
      line_len = line_len + 1;
    }
    line[line_len-1] = '\0';
    err_fn = (char *)malloc(line_len*sizeof(char));
    strcpy(err_fn,line);
  }
  else{
    exit(EXIT_FAILURE);
  }

  fclose(recon_info);
  printf("%s\n", "Data import complete");

  Data_3d recon;
  Data_3d err;
  calc_reconstruction(&recon, &angles, &projs, &err, &param);
  export_3d_data(recon_fn, &recon);
  export_3d_data(err_fn, &err);

  printf("%s\n", "Data export complete");

  free(recon_fn);
  free(err_fn);
  free_2d_data(&angles);
  free_3d_data(&projs);
  free_3d_data(&recon);
  free_3d_data(&err);

  return 0;
}
