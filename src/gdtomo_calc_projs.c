#include <data/data_types.h>
#include <data/file_io.h>
#include <data/data_ops.h>
#include <reconstructor/calc_projs.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int run_gdtomo_calc_projs(const char* calc_projs_info_fn){
  calc_projs_info_fn = calc_projs_info_fn;
  FILE *calc_projs_info;
  char *line;
  size_t len = 0;
  unsigned int line_len = 0;
  char *projs_fn;

  calc_projs_info = fopen(calc_projs_info_fn, "r");
  if(calc_projs_info == NULL)
    exit(EXIT_FAILURE);
  
  Data_3d vol;
  line_len = 0;
  if(getline(&line, &len, calc_projs_info) != -1){
    while(line[line_len] != 0){
      line_len = line_len + 1;
    }
    line[line_len-1] = '\0';
    import_3d_data(line, &vol);
  }
  else{
    exit(EXIT_FAILURE);
  }

  Data_2d angles;
  line_len = 0;
  if(getline(&line, &len, calc_projs_info) != -1){
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

  line_len = 0;
  if(getline(&line, &len, calc_projs_info) != -1){
    while(line[line_len] != 0){
      line_len = line_len + 1;
    }
    line[line_len-1] = '\0';
    projs_fn = (char *)malloc(line_len*sizeof(char));
    strcpy(projs_fn,line);
  }
  else{
    exit(EXIT_FAILURE);
  }

  fclose(calc_projs_info);
  printf("%s\n", "Data import complete");

  Data_3d projs;
  unsigned int *projs_dim = (unsigned int *)malloc(3*sizeof(unsigned int));
  projs_dim[0] = (angles.dim)[0];
  projs_dim[1] = (vol.dim)[0];
  projs_dim[2] = (vol.dim)[1];
  alloc_3d_data(&projs, projs_dim);

  calc_projections(&vol, &angles, &projs);

  export_3d_data(projs_fn, &projs);

  printf("%s\n", "Data export complete");

  free(projs_fn);
  free_3d_data(&vol);
  free_2d_data(&angles);
  free_3d_data(&projs);

  return 0;
}
