#include <reconstructor/calc_projs_1/get_chunk.h>

#include <stdio.h>

double chunk_val(int x, int y, int z, unsigned int x_lim,
  unsigned int y_lim, unsigned int z_lim, double*** vol){
  unsigned int x_idx, y_idx, z_idx;
  if(x<0 || x>(int)x_lim-1){
    return 0.;
  } else{
    x_idx = (unsigned int)x;
  }
  if(y<0 || y>(int)y_lim-1){
    return 0.;
  } else{
    y_idx = (unsigned int)y;
  }
  if(z<0 || z>(int)z_lim-1){
    return 0.;
  } else{
    z_idx = (unsigned int)z;
  }
  return vol[x_idx][y_idx][z_idx];
}

int get_chunk(Data_3d* vol, unsigned int chunk_idx, double* chunk,
  unsigned int dim_chunk, double* chunk_origin){

  int curr_x = -2, curr_y = -2, curr_z = -2;
  unsigned int dim_x = (vol->dim)[0], dim_y = (vol->dim)[1], dim_z = (vol->dim)[2];
  unsigned int num_x = 1, num_y = 1, num_z = 1;
  while(curr_x+dim_chunk < dim_x+2){
    num_x++;
    curr_x = curr_x+dim_chunk-2;
  }
  while(curr_y+dim_chunk < dim_y+2){
    num_y++;
    curr_y = curr_y+dim_chunk-2;
  }
  while(curr_z+dim_chunk < dim_z+2){
    num_z++;
    curr_z = curr_z+dim_chunk-2;
  }

  if(chunk_idx >= num_x*num_y*num_z){
    return 1;
  }

  unsigned int curr_chunk_idx = 0;
  for(unsigned int idx_z = 0; idx_z < num_z; idx_z++){
    for(unsigned int idx_y = 0; idx_y < num_y; idx_y++){
      for(unsigned int idx_x = 0; idx_x < num_x; idx_x++){
        if(curr_chunk_idx == chunk_idx){
          curr_x = -2+(dim_chunk-2)*idx_x;
          curr_y = -2+(dim_chunk-2)*idx_y;
          curr_z = -2+(dim_chunk-2)*idx_z;
        }
        curr_chunk_idx++;
      }
    }
  }

  int x_min = -1*(int)((vol->dim)[0]/2);
  int y_min = -1*(int)((vol->dim)[1]/2);
  int z_min = -1*(int)((vol->dim)[2]/2);
  chunk_origin[0] = (double)(x_min+curr_x);
  chunk_origin[1] = (double)(y_min+curr_y);
  chunk_origin[2] = (double)(z_min+curr_z);

  for(int z = 0; z < (int)dim_chunk; z++){
    for(int y = 0; y < (int)dim_chunk; y++){
      for(int x = 0; x < (int)dim_chunk; x++){
        chunk[z*dim_chunk*dim_chunk+y*dim_chunk+x] =
          chunk_val(
            curr_x+x,curr_y+y,curr_z+z,dim_x,dim_y,dim_z,(vol->data)
          );
      }
    }
  }

  return 0;
}

