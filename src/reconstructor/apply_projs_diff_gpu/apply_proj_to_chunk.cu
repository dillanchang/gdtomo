extern "C"{
  #include <reconstructor/apply_projs_diff_gpu/apply_proj_to_chunk.h>

  #include <data/data_ops.h>
  #include <math.h>
  #include <stdio.h>
}

#define BLOCK_COUNT 512
#define THREAD_COUNT 512

__device__ int in_bounds_gpu(int i, int Ni){
  return (i>=0 && i<Ni);
}
__device__ double proj_interp_val_gpu(double* projs_diff, unsigned int p_idx,
  unsigned int Ni, unsigned int Nj, double x, double y){
  double x_0 = x-floor(x);
  double y_0 = y-floor(y);
  int i = floor(x);
  int j = floor(y);
  double q11, q12, q21, q22;
  if(in_bounds_gpu(i,Ni) && in_bounds_gpu(j,Nj)){
    q11 = projs_diff[p_idx*Nj*Ni+j*Ni+i]; } else{ q11 = 0.;
  }
  if(in_bounds_gpu(i+1,Ni) && in_bounds_gpu(j,Nj)){
    q21 = projs_diff[p_idx*Nj*Ni+j*Ni+(i+1)]; } else{ q21 = 0.;
  }
  if(in_bounds_gpu(i,Ni) && in_bounds_gpu(j+1,Nj)){
    q12 = projs_diff[p_idx*Nj*Ni+(j+1)*Ni+i]; } else{ q12 = 0.;
  }
  if(in_bounds_gpu(i+1,Ni) && in_bounds_gpu(j+1,Nj)){
    q22 = projs_diff[p_idx*Nj*Ni+(j+1)*Ni+(i+1)]; } else{ q22 = 0.;
  }
  double r1 = (1-x_0)*q11+x_0*q21;
  double r2 = (1-x_0)*q12+x_0*q22;
  return (1-y_0)*r1+y_0*r2;
}

__global__ void update_chunk_val(double* projs_diff, unsigned int n_proj,
  unsigned int pdx, unsigned int pdy, double* chunk, unsigned int dim_chunk,
  unsigned int x0, unsigned int y0, unsigned int z0,
  unsigned int vdx, unsigned int vdy, unsigned int vdz, double* r_hats){

  unsigned int tid_shift = 0;                             // device_idx*dim_chunk*dim_chunk*dim_chunk/num_devices;
  unsigned int tid_max   = dim_chunk*dim_chunk*dim_chunk; // (device_idx+1)*dim_chunk*dim_chunk*dim_chunk/num_devices;
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x + tid_shift;

  while(tid < tid_max){
    int i, j, k, x_center, y_center, z_center, p_x_center, p_y_center;
    double x, y, z, rx, ry, v;
    k = tid/(dim_chunk*dim_chunk);
    j = (tid-k*dim_chunk*dim_chunk)/dim_chunk;
    i = tid-k*dim_chunk*dim_chunk-j*dim_chunk;
    z_center = vdz/2;
    y_center = vdy/2;
    x_center = vdx/2;
    z = (double)(k+(int)z0-z_center);
    y = (double)(j+(int)y0-y_center);
    x = (double)(i+(int)x0-x_center);

    p_x_center = pdx/2;
    p_y_center = pdy/2;
    v = 0.;
    for(unsigned int p_idx=0; p_idx<n_proj; p_idx++){
      rx = x*r_hats[p_idx*9+0*3+0]
         + y*r_hats[p_idx*9+0*3+1]
         + z*r_hats[p_idx*9+0*3+2]
         + p_x_center;
      ry = x*r_hats[p_idx*9+1*3+0]
         + y*r_hats[p_idx*9+1*3+1]
         + z*r_hats[p_idx*9+1*3+2]
         + p_y_center;
      v = v + proj_interp_val_gpu(projs_diff, p_idx, pdx, pdy, rx, ry);
    }
    chunk[tid] = v/vdz/n_proj;
    tid += blockDim.x * gridDim.x;
  }

}

void apply_proj_to_chunk(double* dev_projs_diff, unsigned int n_proj,
  unsigned int pdx, unsigned int pdy, double* dev_chunk, unsigned int dim_chunk,
  unsigned int x0, unsigned int y0, unsigned int z0,
  unsigned int vdx, unsigned int vdy, unsigned int vdz, double* dev_r_hats){

  update_chunk_val<<<BLOCK_COUNT,THREAD_COUNT>>>(
    dev_projs_diff, n_proj, pdx, pdy, dev_chunk, dim_chunk,
    x0, y0, z0, vdx, vdy, vdz, dev_r_hats
  );

}

