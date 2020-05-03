#include <reconstructor/calc_projs_1/calc_projs_1.h>
#include <reconstructor/calc_projs_1/get_chunk.h>
#include <reconstructor/calc_projs_1/apply_chunk_to_proj.h>

#include <data/data_ops.h>
#include <stdlib.h>
#include <math.h>

void calc_projs_1(Data_3d* projs, Data_3d* vol, Data_2d* angles){
  unsigned int dim_chunk = 25;
  double*** chunk;
  chunk = (double***)malloc(dim_chunk*sizeof(double**));
  for(unsigned int i=0; i<dim_chunk; i++){
    chunk[i] = (double**)malloc(dim_chunk*sizeof(double*));
    for(unsigned int j=0; j<dim_chunk; j++){
      chunk[i][j] = (double*)malloc(dim_chunk*sizeof(double));
    }
  }
  double *chunk_origin = (double *)malloc(3*sizeof(double));

  double x_hat_i[3] = {1,0,0};
  double y_hat_i[3] = {0,1,0};
  double z_hat_i[3] = {0,0,1};
  double*** r_hats = (double ***)malloc((projs->dim)[0]*sizeof(double **));
  for(unsigned int proj_i = 0; proj_i < (projs->dim)[0]; proj_i++){
    r_hats[proj_i] = (double **)malloc(3*sizeof(double *));
    for(unsigned int i = 0; i < 3; i++){
      r_hats[proj_i][i] = (double *)malloc(3*sizeof(double));
    }
    euler_rot_rev(x_hat_i,(angles->data)[proj_i],r_hats[proj_i][0]);
    euler_rot_rev(y_hat_i,(angles->data)[proj_i],r_hats[proj_i][1]);
    euler_rot_rev(z_hat_i,(angles->data)[proj_i],r_hats[proj_i][2]);
  }

  for(unsigned int proj_idx=0; proj_idx<(projs->dim)[0]; proj_idx++){
    for(unsigned int proj_x=0; proj_x<(projs->dim)[1]; proj_x++){
      for(unsigned int proj_y=0; proj_y<(projs->dim)[2]; proj_y++){
        (projs->data)[proj_idx][proj_x][proj_y] = 0.;
      }
    }
  }

  double x_min  = 1.0*(vol->dim)[0]/2;
  double y_min  = 1.0*(vol->dim)[1]/2;
  double z_min  = 1.0*(vol->dim)[2]/2;
  unsigned int lim_proj_z =
    (unsigned int) ceil(sqrt( (double)x_min*x_min
                            + (double)y_min*y_min
                            + (double)z_min*z_min ));

  unsigned int chunk_idx = 0;
  int finished = get_chunk(vol, chunk_idx, chunk, dim_chunk, chunk_origin);
  while(!finished){
    apply_chunk_to_proj(chunk, dim_chunk, chunk_origin, projs->data,
      r_hats, (projs->dim)[0], (projs->dim)[1], (projs->dim)[2], lim_proj_z);
    chunk_idx++;
    finished = get_chunk(vol, chunk_idx, chunk, dim_chunk, chunk_origin);
  }

  for(unsigned int i=0; i<dim_chunk; i++){
    for(unsigned int j=0; j<dim_chunk; j++){
      free(chunk[i][j]);
    }
    free(chunk[i]);
  }
  free(chunk);
  free(chunk_origin);

  for(unsigned int proj_i = 0; proj_i < (projs->dim)[0]; proj_i++){
    for(unsigned int i = 0; i < 3; i++){
      free(r_hats[proj_i][i]);
    }
    free(r_hats[proj_i]);
  }
  free(r_hats);

}

