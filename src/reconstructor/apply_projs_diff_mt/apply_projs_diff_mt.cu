extern "C"{
  #include <reconstructor/apply_projs_diff_mt/apply_projs_diff_mt.h>
  #include <reconstructor/apply_projs_diff_mt/apply_proj_to_chunk.h>
  
  #include <data/data_ops.h>
  #include <pthread.h>
  #include <stdlib.h>
  #include <stdio.h>
  #include <math.h>
}

void apply_chunk_to_vol(float* chunk, Data_3d* vol, float alpha,
  unsigned int x0, unsigned int y0, unsigned int z0, unsigned int dim_chunk){

  unsigned int x, y, z;
  for(unsigned int i=0; i<dim_chunk; i++){
    for(unsigned int j=0; j<dim_chunk; j++){
      for(unsigned int k=0; k<dim_chunk; k++){
        x = i+x0; y = j+y0; z = k+z0;
        if(x<(vol->dim)[0] && y<(vol->dim)[1] && z<(vol->dim)[2]){
          (vol->data)[x][y][z] = (vol->data)[x][y][z]
            + alpha*chunk[k*dim_chunk*dim_chunk+j*dim_chunk+i];
        }
      }
    }
  }
}

void apply_projs_diff_mt(Data_3d* vol, Data_3d* projs_diff, Data_2d* angles, 
  float alpha){

  // Get chunk size
  unsigned int vdx = (vol->dim)[0];
  unsigned int vdy = (vol->dim)[1];
  unsigned int vdz = (vol->dim)[2];
  size_t total_mem;
  cudaSetDevice(0);
  cudaMemGetInfo(NULL, &total_mem);
  unsigned int dim_chunk = (unsigned int)pow(total_mem*0.25/sizeof(float),1./3.);
  if(dim_chunk > vdx) dim_chunk = vdx;
  if(dim_chunk > vdy) dim_chunk = vdy;
  if(dim_chunk > vdz) dim_chunk = vdz;

  // Determine number of projection jobs
  unsigned int n_projs_tot = (projs_diff->dim)[0];
  unsigned int pdx = (projs_diff->dim)[1];
  unsigned int pdy = (projs_diff->dim)[2];
  unsigned int n_jobs = 
    (unsigned int)ceil(1.*n_projs_tot*pdx*pdy*sizeof(float)/(total_mem*0.50));

  // Establish projection idxs for each job
  unsigned int* proj_idx_low  = (unsigned int*)malloc(n_jobs*sizeof(unsigned int));
  unsigned int* n_projs       = (unsigned int*)malloc(n_jobs*sizeof(unsigned int));
  for(unsigned int job_idx = 0; job_idx < n_jobs; job_idx++){
    unsigned int inc = ceil(1.0*n_projs_tot/n_jobs);
    proj_idx_low[job_idx]  = job_idx*inc;
    unsigned int proj_idx_high = (job_idx+1)*inc-1;
    if(proj_idx_high > n_projs_tot-1){
      proj_idx_high = n_projs_tot-1;
    }
    n_projs[job_idx] = proj_idx_high - proj_idx_low[job_idx] + 1;
  }

  // Calculate r_hats
  float x_hat_i[3] = {1,0,0};
  float y_hat_i[3] = {0,1,0};
  float z_hat_i[3] = {0,0,1};
  float** r_hats = (float **)malloc(n_jobs*sizeof(float *));
  float v[3];
  for(unsigned int job_idx = 0; job_idx < n_jobs; job_idx++){
    r_hats[job_idx] = (float *)malloc(n_projs[job_idx]*3*3*sizeof(float));
    unsigned int proj_idx;
    for(unsigned int i = 0; i < n_projs[job_idx]; i++){
      proj_idx = proj_idx_low[job_idx] + i;
      euler_rot(x_hat_i,(angles->data)[proj_idx],v);
      r_hats[job_idx][i*9+0*3+0] = v[0];
      r_hats[job_idx][i*9+0*3+1] = v[1];
      r_hats[job_idx][i*9+0*3+2] = v[2];
      euler_rot(y_hat_i,(angles->data)[proj_idx],v);
      r_hats[job_idx][i*9+1*3+0] = v[0];
      r_hats[job_idx][i*9+1*3+1] = v[1];
      r_hats[job_idx][i*9+1*3+2] = v[2];
      euler_rot(z_hat_i,(angles->data)[proj_idx],v);
      r_hats[job_idx][i*9+2*3+0] = v[0];
      r_hats[job_idx][i*9+2*3+1] = v[1];
      r_hats[job_idx][i*9+2*3+2] = v[2];
    }
  }

  // Allocate proj_diff_arrs
  float** proj_diff_arrs = (float **)malloc(n_jobs*sizeof(float *));
  for(unsigned int job_idx = 0; job_idx < n_jobs; job_idx++){
    proj_diff_arrs[job_idx] = (float *)malloc(n_projs[job_idx]*pdy*pdx*sizeof(float));
    for(unsigned int i = 0; i < n_projs[job_idx]; i++){
      unsigned int proj_idx = proj_idx_low[job_idx]+i;
      for(unsigned int px = 0; px < pdx; px++){
        for(unsigned int py = 0; py < pdy; py++){
          proj_diff_arrs[job_idx][i*pdy*pdx+py*pdx+px] = (projs_diff->data)[proj_idx][px][py];
        }
      }
    }
  }

  // Allocate chunks
  float* chunk = (float *)malloc(dim_chunk*dim_chunk*dim_chunk*sizeof(float));
  float *dev_chunk, *dev_proj_diff, *dev_r_hats;

  for(unsigned int job_idx = 0; job_idx < n_jobs; job_idx++){
    cudaMalloc( (void**)&(dev_proj_diff), n_projs[job_idx]*pdx*pdy*sizeof(float)      );
    cudaMalloc( (void**)&(dev_r_hats),    n_projs[job_idx]*3*3*sizeof(float)          );
    cudaMalloc( (void**)&(dev_chunk),     dim_chunk*dim_chunk*dim_chunk*sizeof(float) );
    cudaMemcpy( dev_proj_diff, 
                proj_diff_arrs[job_idx],
                n_projs[job_idx]*pdx*pdy*sizeof(float),
                cudaMemcpyHostToDevice );
    cudaMemcpy( dev_r_hats,
                r_hats[job_idx],
                n_projs[job_idx]*3*3*sizeof(float),
                cudaMemcpyHostToDevice );

    for(unsigned int z0=0; z0<vdz; z0=z0+dim_chunk){
      for(unsigned int y0=0; y0<vdy; y0=y0+dim_chunk){
        for(unsigned int x0=0; x0<vdx; x0=x0+dim_chunk){
          apply_proj_to_chunk(dev_proj_diff, n_projs[job_idx], n_projs_tot, pdx, pdy, dev_chunk,
            dim_chunk, x0, y0, z0, vdx, vdy, vdz, dev_r_hats);
          cudaMemcpy( chunk,
                      dev_chunk,
                      dim_chunk*dim_chunk*dim_chunk*sizeof(float),
                      cudaMemcpyDeviceToHost );
          apply_chunk_to_vol(chunk, vol, alpha, x0, y0, z0, dim_chunk);
        }
      }
    }
    cudaFree(dev_chunk);
    cudaFree(dev_proj_diff);
    cudaFree(dev_r_hats);
  }

  free(chunk);
  free(proj_idx_low);
  free(n_projs);
  for(unsigned int job_idx = 0; job_idx < n_jobs; job_idx++){
    free(r_hats[job_idx]);
    free(proj_diff_arrs[job_idx]);
  }
  free(proj_diff_arrs);
  free(r_hats);

}

