extern "C"{
  #include <reconstructor/calc_projs_gpu/calc_projs_gpu.h>
  #include <reconstructor/calc_projs_gpu/apply_chunk_to_proj.h>
  #include <reconstructor/calc_projs_gpu/get_chunk.h>
  
  #include <data/data_ops.h>
  #include <pthread.h>
  #include <stdlib.h>
  #include <stdio.h>
  #include <math.h>
}

struct PC_Struct {
  double*       proj_arr;
  double*       r_hats;
  Data_3d*      vol;
  int           device_idx;
  unsigned int  dim_chunk;
  unsigned int  n_proj;
  unsigned int  pdx;
  unsigned int  pdy;
  double        lim_proj_z;
};

void* exec_proj_job(void *void_data){
  PC_Struct *data = (PC_Struct *)void_data;
  double*       proj_arr   = data->proj_arr  ;
  double*       r_hats     = data->r_hats    ;
  Data_3d*      vol        = data->vol       ;
  int           device_idx = data->device_idx;
  unsigned int  dim_chunk  = data->dim_chunk ;
  unsigned int  n_proj     = data->n_proj    ;
  unsigned int  pdx        = data->pdx       ;
  unsigned int  pdy        = data->pdy       ;
  double        lim_proj_z = data->lim_proj_z;

  cudaSetDevice(device_idx);
  double* chunk = (double*)malloc(dim_chunk*dim_chunk*dim_chunk*sizeof(double));
  double* chunk_origin = (double*)malloc(3*sizeof(double));
  double *dev_chunk, *dev_chunk_origin, *dev_projs, *dev_r_hats;
  cudaMalloc( (void**)&dev_chunk, dim_chunk*dim_chunk*dim_chunk*sizeof(double) );
  cudaMalloc( (void**)&dev_chunk_origin, 3*sizeof(double) );
  cudaMalloc( (void**)&dev_projs, n_proj*pdx*pdy*sizeof(double) );
  cudaMalloc( (void**)&dev_r_hats, n_proj*3*3*sizeof(double) );
  cudaMemcpy( dev_projs, 
              proj_arr,
              n_proj*pdx*pdy*sizeof(double),
              cudaMemcpyHostToDevice );
  cudaMemcpy( dev_r_hats, 
              r_hats,
              n_proj*3*3*sizeof(double),
              cudaMemcpyHostToDevice );

  unsigned int chunk_idx = 0;
  int finished = get_chunk(vol, chunk_idx, chunk, dim_chunk, chunk_origin);
  while(!finished){
    cudaMemcpy( dev_chunk, 
                chunk,
                dim_chunk*dim_chunk*dim_chunk*sizeof(double),
                cudaMemcpyHostToDevice );
    cudaMemcpy( dev_chunk_origin, 
                chunk_origin,
                3*sizeof(double),
                cudaMemcpyHostToDevice );
    apply_chunk_to_proj(dev_chunk, dim_chunk, dev_chunk_origin, dev_projs,
      dev_r_hats, n_proj, pdx, pdy, lim_proj_z);
    chunk_idx++;
    finished = get_chunk(vol, chunk_idx, chunk, dim_chunk, chunk_origin);
  }

  cudaMemcpy( proj_arr, 
              dev_projs,
              n_proj*pdx*pdy*sizeof(double),
              cudaMemcpyDeviceToHost );
  cudaFree( dev_chunk        );
  cudaFree( dev_chunk_origin );
  cudaFree( dev_projs        );
  cudaFree( dev_r_hats       );
  free(chunk       );
  free(chunk_origin);

  return 0;
}

void calc_projs_gpu(Data_3d* projs, Data_3d* vol, Data_2d* angles){
  
  // Get device count
  int n_devices;
  cudaGetDeviceCount(&n_devices);

  // Get chunk size based on GPU with smallest memory
  size_t free_mem;
  cudaSetDevice(0);
  cudaMemGetInfo(&free_mem, NULL);
  for(int device_idx = 1; device_idx < n_devices; device_idx++){
    size_t free_mem_idx;
    cudaSetDevice(device_idx);
    cudaMemGetInfo(&free_mem_idx, NULL);
    if(free_mem_idx < free_mem){
      free_mem = free_mem_idx;
    }
  }
  unsigned int dim_chunk = (unsigned int)pow(1.0*(free_mem*0.25)/sizeof(double),1./3.);
  if(dim_chunk > (vol->dim)[0]+4) dim_chunk = (vol->dim)[0]+4;
  if(dim_chunk > (vol->dim)[1]+4) dim_chunk = (vol->dim)[1]+4;
  if(dim_chunk > (vol->dim)[2]+4) dim_chunk = (vol->dim)[2]+4;

  // Determine number of projection jobs
  unsigned int n_projs_tot = (projs->dim)[0];
  unsigned int pdx = (projs->dim)[1];
  unsigned int pdy = (projs->dim)[2];
  unsigned int n_jobs = 
    (unsigned int)ceil(1.*pdx*pdy*sizeof(double)/(free_mem*0.25));
  if(n_jobs < n_devices){
    n_jobs = n_devices;
  }
  n_jobs = 3; // TESTING HERE [DO NOT FORGET]

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
  double x_hat_i[3] = {1,0,0};
  double y_hat_i[3] = {0,1,0};
  double z_hat_i[3] = {0,0,1};
  double** r_hats = (double **)malloc(n_jobs*sizeof(double *));
  double v[3];
  for(unsigned int job_idx = 0; job_idx < n_jobs; job_idx++){
    r_hats[job_idx] = (double *)malloc(n_projs[job_idx]*3*3*sizeof(double));
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
  double x_min  = 1.0*(vol->dim)[0]/2;
  double y_min  = 1.0*(vol->dim)[1]/2;
  double z_min  = 1.0*(vol->dim)[2]/2;
  unsigned int lim_proj_z =
    (unsigned int) ceil(sqrt( (double)x_min*x_min
                            + (double)y_min*y_min
                            + (double)z_min*z_min ));

  // Allocate proj_arrs
  double** proj_arrs = (double **)malloc(n_jobs*sizeof(double *));
  for(unsigned int job_idx = 0; job_idx < n_jobs; job_idx++){
    proj_arrs[job_idx] = (double *)malloc(n_projs[job_idx]*pdy*pdx*sizeof(double));
    for(unsigned int idx=0; idx<n_projs[job_idx]*pdy*pdx; idx++){
      proj_arrs[job_idx][idx] = 0.;
    }
  }

  // Perform main algorithm
  PC_Struct* data   = (PC_Struct *)malloc(n_devices*sizeof(PC_Struct));
  pthread_t* threads = (pthread_t *)malloc(n_devices*sizeof(pthread_t));
  unsigned int n_cycles = (unsigned int)ceil(1.0*n_jobs/n_devices);
  for(unsigned int cycle = 0; cycle < n_cycles; cycle++){
    unsigned int job_idx;
    for(unsigned int device_idx = 0; device_idx < n_devices; device_idx++){
      job_idx = cycle*n_devices + device_idx;
      if(job_idx < n_jobs){
        data[device_idx].proj_arr   = proj_arrs[job_idx];
        data[device_idx].r_hats     = r_hats[job_idx];
        data[device_idx].vol        = vol;
        data[device_idx].device_idx = device_idx;
        data[device_idx].dim_chunk  = dim_chunk;
        data[device_idx].n_proj     = n_projs[job_idx];
        data[device_idx].pdx        = pdx;
        data[device_idx].pdy        = pdy;
        data[device_idx].lim_proj_z = lim_proj_z;
        pthread_create(&(threads[device_idx]), NULL, exec_proj_job, &(data[device_idx]));
      }
    }
    for(unsigned int device_idx = 0; device_idx < n_devices; device_idx++){
      job_idx = cycle*n_devices + device_idx;
      if(job_idx < n_jobs){
        pthread_join(threads[device_idx], NULL);
      }
    }
    for(unsigned int device_idx = 0; device_idx < n_devices; device_idx++){
      job_idx = cycle*n_devices + device_idx;
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

