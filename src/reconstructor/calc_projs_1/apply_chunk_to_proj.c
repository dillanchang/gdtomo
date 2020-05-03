#include <reconstructor/calc_projs_1/apply_chunk_to_proj.h>

#include <data/data_ops.h>
#include <math.h>

int in_chunk(double x, double y, double z, unsigned int dim_chunk){
  double low = 0.5, high = (double)(dim_chunk-1)-.5;
  if(x < low || x >= high){
    return 0;
  }
  if(y < low || y >= high){
    return 0;
  }
  if(z < low || z >= high){
    return 0;
  }
  return 1;
}

double interp3(double x, double y, double z, double*** vol){
  double x_0 = x-floor(x); int i = (int)floor(x);
  double y_0 = y-floor(y); int j = (int)floor(y);
  double z_0 = z-floor(z); int k = (int)floor(z);
  
  double q000, q100, q010, q001, q110, q101, q011, q111;
  q000 = vol[i][j][k];
  q100 = vol[i+1][j][k];
  q010 = vol[i][j+1][k];
  q001 = vol[i][j][k+1];
  q110 = vol[i+1][j+1][k];
  q101 = vol[i+1][j][k+1];
  q011 = vol[i][j+1][k+1];
  q111 = vol[i+1][j+1][k+1];
  return  q000*(1-x_0)*(1-y_0)*(1-z_0) +
          q100*x_0*(1-y_0)*(1-z_0)     +
          q010*(1-x_0)*y_0*(1-z_0)     +
          q001*(1-x_0)*(1-y_0)*z_0     +
          q110*x_0*y_0*(1-z_0)         +
          q101*x_0*(1-y_0)*z_0         +
          q011*(1-x_0)*y_0*z_0         +
          q111*x_0*y_0*z_0;
}

void get_proj_val(double*** chunk, unsigned int dim_chunk,
  double* chunk_origin, double*** projs, double*** r_hats,
  unsigned int dim_proj_x, unsigned int dim_proj_y, unsigned int lim_proj_z,
  unsigned int proj_idx, unsigned int proj_x, unsigned int proj_y){

  double x, y, z, x_p, y_p, z_p, v;
  x_p = (double)((int)proj_x-(int)(dim_proj_x/2));
  y_p = (double)((int)proj_y-(int)(dim_proj_y/2));
  v = 0.;
  for(int k_p = -1*(int)(lim_proj_z); k_p <= (int)(lim_proj_z); k_p++){
    z_p = (double)k_p;
    x = x_p*r_hats[proj_idx][0][0]
      + y_p*r_hats[proj_idx][0][1]
      + z_p*r_hats[proj_idx][0][2]
      - chunk_origin[0]; 
    y = x_p*r_hats[proj_idx][1][0]
      + y_p*r_hats[proj_idx][1][1]
      + z_p*r_hats[proj_idx][1][2]
      - chunk_origin[1]; 
    z = x_p*r_hats[proj_idx][2][0]
      + y_p*r_hats[proj_idx][2][1]
      + z_p*r_hats[proj_idx][2][2]
      - chunk_origin[2]; 
    if(in_chunk(x, y, z, dim_chunk)){
      v = v + interp3(x, y, z, chunk);
    }
  }
  projs[proj_idx][proj_x][proj_y] = projs[proj_idx][proj_x][proj_y] + v;
}

void apply_chunk_to_proj(double*** chunk, unsigned int dim_chunk,
  double* chunk_origin, double*** projs, double*** r_hats,
  unsigned int num_projs, unsigned int dim_proj_x, unsigned int dim_proj_y,
  unsigned int lim_proj_z){

  for(unsigned int proj_idx = 0; proj_idx < num_projs; proj_idx++){
    for(unsigned int proj_x = 0; proj_x < dim_proj_x; proj_x++){
      for(unsigned int proj_y = 0; proj_y < dim_proj_y; proj_y++){
        get_proj_val(chunk, dim_chunk, chunk_origin, projs, r_hats,
          dim_proj_x, dim_proj_y, lim_proj_z, proj_idx, proj_x, proj_y);
      }
    }
  }

}

