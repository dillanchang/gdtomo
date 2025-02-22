extern "C"{
  #include <reconstructor/calc_projs_mt/calc_projs_mt.h>
  #include <reconstructor/calc_projs_mt/apply_chunk_to_proj.h>
  #include <reconstructor/calc_projs_mt/get_chunk.h>
  
  #include <data/data_ops.h>
  #include <pthread.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <math.h>
}

void exec_proj_job(float* proj_arr, float* r_hats, Data_3d* vol,
  unsigned int dim_chunk, unsigned int n_proj, unsigned int pdx,
  unsigned int pdy, float lim_proj_z){
  
  float* chunk = (float*)malloc(dim_chunk*dim_chunk*dim_chunk*sizeof(float));
  float* chunk_origin = (float*)malloc(3*sizeof(float));
  float *dev_chunk, *dev_chunk_origin, *dev_projs, *dev_r_hats;
  cudaMalloc( (void**)&dev_chunk, dim_chunk*dim_chunk*dim_chunk*sizeof(float) );
  cudaMalloc( (void**)&dev_chunk_origin, 3*sizeof(float) );
  cudaMalloc( (void**)&dev_projs, n_proj*pdx*pdy*sizeof(float) );
  cudaMalloc( (void**)&dev_r_hats, n_proj*3*3*sizeof(float) );
  cudaMemcpy( dev_projs, 
              proj_arr,
              n_proj*pdx*pdy*sizeof(float),
              cudaMemcpyHostToDevice );
  cudaMemcpy( dev_r_hats, 
              r_hats,
              n_proj*3*3*sizeof(float),
              cudaMemcpyHostToDevice );

  unsigned int chunk_idx = 0;
  int finished = get_chunk(vol, chunk_idx, chunk, dim_chunk, chunk_origin);
  while(!finished){
    cudaMemcpy( dev_chunk, 
                chunk,
                dim_chunk*dim_chunk*dim_chunk*sizeof(float),
                cudaMemcpyHostToDevice );
    cudaMemcpy( dev_chunk_origin, 
                chunk_origin,
                3*sizeof(float),
                cudaMemcpyHostToDevice );
    apply_chunk_to_proj(dev_chunk, dim_chunk, dev_chunk_origin, dev_projs,
      dev_r_hats, n_proj, pdx, pdy, lim_proj_z);
    chunk_idx++;
    finished = get_chunk(vol, chunk_idx, chunk, dim_chunk, chunk_origin);
  }

  cudaMemcpy( proj_arr, 
              dev_projs,
              n_proj*pdx*pdy*sizeof(float),
              cudaMemcpyDeviceToHost );
  cudaFree( dev_chunk        );
  cudaFree( dev_chunk_origin );
  cudaFree( dev_projs        );
  cudaFree( dev_r_hats       );
  free(chunk       );
  free(chunk_origin);

}

void calc_projs_mt(Data_3d* projs, Data_3d* vol, Data_2d* angles){
  
  // Get chunk size based on GPU with smallest memory
  size_t total_mem;
  cudaSetDevice(0);
  cudaMemGetInfo(NULL, &total_mem);
  unsigned int dim_chunk = (unsigned int)pow(1.0*(total_mem*0.25)/sizeof(float),1./3.);
  if(dim_chunk > (vol->dim)[0]+4) dim_chunk = (vol->dim)[0]+4;
  if(dim_chunk > (vol->dim)[1]+4) dim_chunk = (vol->dim)[1]+4;
  if(dim_chunk > (vol->dim)[2]+4) dim_chunk = (vol->dim)[2]+4;

  // Determine number of projection jobs
  unsigned int n_projs_tot = (projs->dim)[0];
  unsigned int pdx = (projs->dim)[1];
  unsigned int pdy = (projs->dim)[2];
  unsigned int n_jobs = 
    (unsigned int)ceil(1.*n_projs_tot*pdx*pdy*sizeof(float)/(total_mem*0.25));

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
      euler_rot_rev(x_hat_i,(angles->data)[proj_idx],v);
      r_hats[job_idx][i*9+0*3+0] = v[0];
      r_hats[job_idx][i*9+0*3+1] = v[1];
      r_hats[job_idx][i*9+0*3+2] = v[2];
      euler_rot_rev(y_hat_i,(angles->data)[proj_idx],v);
      r_hats[job_idx][i*9+1*3+0] = v[0];
      r_hats[job_idx][i*9+1*3+1] = v[1];
      r_hats[job_idx][i*9+1*3+2] = v[2];
      euler_rot_rev(z_hat_i,(angles->data)[proj_idx],v);
      r_hats[job_idx][i*9+2*3+0] = v[0];
      r_hats[job_idx][i*9+2*3+1] = v[1];
      r_hats[job_idx][i*9+2*3+2] = v[2];
    }
  }
  float x_min  = 1.0*(vol->dim)[0]/2;
  float y_min  = 1.0*(vol->dim)[1]/2;
  float z_min  = 1.0*(vol->dim)[2]/2;
  unsigned int lim_proj_z =
    (unsigned int) ceil(sqrt( (float)x_min*x_min
                            + (float)y_min*y_min
                            + (float)z_min*z_min ));

  // Allocate proj_arrs
  float** proj_arrs = (float **)malloc(n_jobs*sizeof(float *));
  for(unsigned int job_idx = 0; job_idx < n_jobs; job_idx++){
    proj_arrs[job_idx] = (float *)malloc(n_projs[job_idx]*pdy*pdx*sizeof(float));
    for(unsigned int idx=0; idx<n_projs[job_idx]*pdy*pdx; idx++){
      proj_arrs[job_idx][idx] = 0.;
    }
  }

  // Perform main algorithm
  for(unsigned int job_idx = 0; job_idx < n_jobs; job_idx++){
    exec_proj_job(proj_arrs[job_idx],r_hats[job_idx],vol,dim_chunk,n_projs[job_idx],pdx,
        pdy, lim_proj_z);
    for(unsigned int i = 0; i < n_projs[job_idx]; i++){
      unsigned int proj_idx = proj_idx_low[job_idx]+i;
      for(unsigned int py = 0; py < pdy; py++){
        for(unsigned int px = 0; px < pdx; px++){
          (projs->data)[proj_idx][px][py] =
            proj_arrs[job_idx][i*pdy*pdx+py*pdx+px];
        }
      }
    }
  }

  // Free mallocs
  free(proj_idx_low);
  free(n_projs);
  for(unsigned int job_idx = 0; job_idx < n_jobs; job_idx++){
    free(r_hats[job_idx]);
    free(proj_arrs[job_idx]);
  }
  free(proj_arrs);
  free(r_hats);

}

