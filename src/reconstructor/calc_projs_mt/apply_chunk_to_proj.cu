extern "C"{
  #include <reconstructor/calc_projs_mt/apply_chunk_to_proj.h>

  #include <data/data_ops.h>
  #include <math.h>
}

#define BLOCK_COUNT 512
#define THREAD_COUNT 512

__device__ int in_chunk(float x, float y, float z, unsigned int dim_chunk){
  float low = 0.5, high = (float)(dim_chunk-1)-.5;
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

__device__ float interp3(float x, float y, float z, float* chunk, unsigned int d){
  float x_0 = x-floor(x); int i = (int)floor(x);
  float y_0 = y-floor(y); int j = (int)floor(y);
  float z_0 = z-floor(z); int k = (int)floor(z);
  
  float q000, q100, q010, q001, q110, q101, q011, q111;
  q000 = chunk[k*d*d+j*d+i];
  q100 = chunk[k*d*d+j*d+(i+1)];
  q010 = chunk[k*d*d+(j+1)*d+i];
  q001 = chunk[(k+1)*d*d+j*d+i];
  q110 = chunk[k*d*d+(j+1)*d+(i+1)];
  q101 = chunk[(k+1)*d*d+j*d+(i+1)];
  q011 = chunk[(k+1)*d*d+(j+1)*d+i];
  q111 = chunk[(k+1)*d*d+(j+1)*d+(i+1)];
  return  q000*(1-x_0)*(1-y_0)*(1-z_0) +
          q100*x_0*(1-y_0)*(1-z_0)     +
          q010*(1-x_0)*y_0*(1-z_0)     +
          q001*(1-x_0)*(1-y_0)*z_0     +
          q110*x_0*y_0*(1-z_0)         +
          q101*x_0*(1-y_0)*z_0         +
          q011*(1-x_0)*y_0*z_0         +
          q111*x_0*y_0*z_0;
}

__global__ void get_proj_val(float* chunk, unsigned int dim_chunk,
  float* chunk_origin, float* projs, float* r_hats,
  unsigned int n_proj, unsigned int pdx, unsigned int pdy, unsigned int lim_proj_z){

  unsigned int tid_shift = 0; // device_idx*n_proj*pdx*pdy/num_devices;
  unsigned int tid_max   = n_proj*pdx*pdy; // (device_idx+1)*n_proj*pdx*pdy/num_devices;
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x + tid_shift;

  while(tid < tid_max){
    unsigned int p_idx = tid/(pdy*pdx);
    unsigned int py    = (tid-p_idx*pdy*pdx)/pdx;
    unsigned int px    = tid-p_idx*pdy*pdx-py*pdx;

    float x, y, z, x_p, y_p, z_p, v;
    x_p = (float)((int)px-(int)(pdx/2));
    y_p = (float)((int)py-(int)(pdy/2));
    v = 0.;
    for(int k_p = -1*(int)(lim_proj_z); k_p <= (int)(lim_proj_z); k_p++){
      z_p = (float)k_p;
      x = x_p*r_hats[p_idx*9+0*3+0]
        + y_p*r_hats[p_idx*9+0*3+1]
        + z_p*r_hats[p_idx*9+0*3+2]
        - chunk_origin[0]; 
      y = x_p*r_hats[p_idx*9+1*3+0]
        + y_p*r_hats[p_idx*9+1*3+1]
        + z_p*r_hats[p_idx*9+1*3+2]
        - chunk_origin[1]; 
      z = x_p*r_hats[p_idx*9+2*3+0]
        + y_p*r_hats[p_idx*9+2*3+1]
        + z_p*r_hats[p_idx*9+2*3+2]
        - chunk_origin[2]; 
      if(in_chunk(x, y, z, dim_chunk)){
        v = v + interp3(x, y, z, chunk, dim_chunk);
      }
    }
    projs[tid] = projs[tid] + v;
    tid += blockDim.x * gridDim.x;
  }
}

void apply_chunk_to_proj(float* dev_chunk, unsigned int dim_chunk,
  float* dev_chunk_origin, float* dev_projs, float* dev_r_hats,
  unsigned int num_projs, unsigned int pdx, unsigned int pdy,
  unsigned int lim_proj_z){
  
  get_proj_val<<<BLOCK_COUNT,THREAD_COUNT>>>(dev_chunk, dim_chunk,
    dev_chunk_origin, dev_projs, dev_r_hats, num_projs, pdx, pdy, lim_proj_z
  );

}

