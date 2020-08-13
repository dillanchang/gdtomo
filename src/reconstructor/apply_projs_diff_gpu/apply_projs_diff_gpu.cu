extern "C"{
  #include <reconstructor/apply_projs_diff_gpu/apply_projs_diff_gpu.h>
  #include <reconstructor/apply_projs_diff_gpu/apply_proj_to_chunk.h>
  
  #include <data/data_ops.h>
  #include <pthread.h>
  #include <stdlib.h>
  #include <math.h>
}

struct AP_Struct {
  unsigned int device_idx;
  double*      dev_proj_diff;
  unsigned int n_projs;
  unsigned int n_projs_tot;
  unsigned int pdx;
  unsigned int pdy;
  double*      dev_chunk;
  double*      chunk;
  unsigned int dim_chunk;
  unsigned int x0;
  unsigned int y0;
  unsigned int z0;
  unsigned int vdx;
  unsigned int vdy;
  unsigned int vdz;
  double*      dev_r_hats;      
};

void* exec_apply_job(void *void_data){
  AP_Struct *data = (AP_Struct *)void_data;
  unsigned int device_idx    = data->device_idx;
  double*      dev_proj_diff = data->dev_proj_diff;
  unsigned int n_projs       = data->n_projs;
  unsigned int n_projs_tot   = data->n_projs_tot;
  unsigned int pdx           = data->pdx;
  unsigned int pdy           = data->pdy;
  double*      dev_chunk     = data->dev_chunk;
  double*      chunk         = data->chunk;
  unsigned int dim_chunk     = data->dim_chunk;
  unsigned int x0            = data->x0;
  unsigned int y0            = data->y0;
  unsigned int z0            = data->z0;
  unsigned int vdx           = data->vdx;
  unsigned int vdy           = data->vdy;
  unsigned int vdz           = data->vdz;
  double*      dev_r_hats    = data->dev_r_hats;

  cudaSetDevice(device_idx);
  apply_proj_to_chunk(dev_proj_diff, n_projs, n_projs_tot, pdx, pdy, dev_chunk,
    dim_chunk, x0, y0, z0, vdx, vdy, vdz, dev_r_hats);
  cudaMemcpy( chunk,
              dev_chunk,
              dim_chunk*dim_chunk*dim_chunk*sizeof(double),
              cudaMemcpyDeviceToHost );
  return 0;
}


void apply_chunks_to_vol(double** chunks, Data_3d* vol, double alpha,
  unsigned int x0, unsigned int y0, unsigned int z0, unsigned int n_devices,
  unsigned int dim_chunk){

  unsigned int x, y, z;
  for(unsigned int device_idx = 0; device_idx < n_devices; device_idx++){
    for(unsigned int i=0; i<dim_chunk; i++){
      for(unsigned int j=0; j<dim_chunk; j++){
        for(unsigned int k=0; k<dim_chunk; k++){
          x = i+x0; y = j+y0; z = k+z0;
          if(x<(vol->dim)[0] && y<(vol->dim)[1] && z<(vol->dim)[2]){
            (vol->data)[x][y][z] = (vol->data)[x][y][z]
              + alpha*chunks[device_idx][k*dim_chunk*dim_chunk+j*dim_chunk+i];
          }
        }
      }
    }
  }
}

void apply_projs_diff_gpu(Data_3d* vol, Data_3d* projs_diff, Data_2d* angles, 
  double alpha){

  // Get device count
  int n_devices;
  cudaGetDeviceCount(&n_devices);

  // Get chunk size
  unsigned int vdx = (vol->dim)[0];
  unsigned int vdy = (vol->dim)[1];
  unsigned int vdz = (vol->dim)[2];
  size_t total_mem;
  cudaSetDevice(0);
  cudaMemGetInfo(NULL, &total_mem);
  unsigned int dim_chunk = (unsigned int)pow(total_mem*0.25/sizeof(double),1./3.);
  if(dim_chunk > vdx) dim_chunk = vdx;
  if(dim_chunk > vdy) dim_chunk = vdy;
  if(dim_chunk > vdz) dim_chunk = vdz;

  // Determine number of projection jobs
  unsigned int n_projs_tot = (projs_diff->dim)[0];
  unsigned int pdx = (projs_diff->dim)[1];
  unsigned int pdy = (projs_diff->dim)[2];
  unsigned int n_jobs = 
    (unsigned int)ceil(1.*n_projs_tot*pdx*pdy*sizeof(double)/(total_mem*0.25));
  if(n_jobs < n_devices){
    n_devices = n_jobs;
  }

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
  double** proj_diff_arrs = (double **)malloc(n_jobs*sizeof(double *));
  for(unsigned int job_idx = 0; job_idx < n_jobs; job_idx++){
    proj_diff_arrs[job_idx] = (double *)malloc(n_projs[job_idx]*pdy*pdx*sizeof(double));
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
  double** chunks = (double**)malloc(n_devices*sizeof(double *));
  for(unsigned int device_idx = 0; device_idx < n_devices; device_idx++){
    chunks[device_idx] = (double *)malloc(dim_chunk*dim_chunk*dim_chunk*sizeof(double));
  }

  double **dev_chunk, **dev_proj_diff, **dev_r_hats;
  dev_proj_diff = (double **)malloc(n_devices*sizeof(double *));
  dev_r_hats    = (double **)malloc(n_devices*sizeof(double *));
  dev_chunk     = (double **)malloc(n_devices*sizeof(double *));

  AP_Struct* data   = (AP_Struct *)malloc(n_devices*sizeof(AP_Struct));
  pthread_t* threads = (pthread_t *)malloc(n_devices*sizeof(pthread_t));

  unsigned int n_cycles = (unsigned int)ceil(1.0*n_jobs/n_devices);
  for(unsigned int cycle = 0; cycle < n_cycles; cycle++){
    unsigned int job_idx;

    for(unsigned int device_idx = 0; device_idx < n_devices; device_idx++){
      cudaSetDevice(device_idx);
      job_idx = cycle*n_devices + device_idx;
      cudaMalloc( (void**)&(dev_proj_diff[device_idx]), n_projs[job_idx]*pdx*pdy*sizeof(double)  );
      cudaMalloc( (void**)&(dev_r_hats[device_idx]), n_projs[job_idx]*3*3*sizeof(double)         );
      cudaMalloc( (void**)&(dev_chunk[device_idx]), dim_chunk*dim_chunk*dim_chunk*sizeof(double) );
      cudaMemcpy( dev_proj_diff[device_idx], 
                  proj_diff_arrs[job_idx],
                  n_projs[job_idx]*pdx*pdy*sizeof(double),
                  cudaMemcpyHostToDevice );
      cudaMemcpy( dev_r_hats[device_idx],
                  r_hats[job_idx],
                  n_projs[job_idx]*3*3*sizeof(double),
                  cudaMemcpyHostToDevice );
    }

    for(unsigned int z0=0; z0<vdz; z0=z0+dim_chunk){
      for(unsigned int y0=0; y0<vdy; y0=y0+dim_chunk){
        for(unsigned int x0=0; x0<vdx; x0=x0+dim_chunk){
          for(unsigned int device_idx = 0; device_idx < n_devices; device_idx++){
            job_idx = cycle*n_devices + device_idx;
            if(job_idx < n_jobs){
              data[device_idx].device_idx    = device_idx;
              data[device_idx].dev_proj_diff = dev_proj_diff[device_idx];
              data[device_idx].n_projs       = n_projs[job_idx];
              data[device_idx].n_projs_tot   = n_projs_tot;
              data[device_idx].pdx           = pdx;
              data[device_idx].pdy           = pdy;
              data[device_idx].dev_chunk     = dev_chunk[device_idx];
              data[device_idx].chunk         = chunks[device_idx];
              data[device_idx].dim_chunk     = dim_chunk;
              data[device_idx].x0            = x0;
              data[device_idx].y0            = y0;
              data[device_idx].z0            = z0;
              data[device_idx].vdx           = vdx;
              data[device_idx].vdy           = vdy;
              data[device_idx].vdz           = vdz;
              data[device_idx].dev_r_hats    = dev_r_hats[device_idx];      
              pthread_create(&(threads[device_idx]), NULL, exec_apply_job, &(data[device_idx]));
            }
          }
          for(unsigned int device_idx = 0; device_idx < n_devices; device_idx++){
            job_idx = cycle*n_devices + device_idx;
            if(job_idx < n_jobs){
              pthread_join(threads[device_idx], NULL);
            }
          }
          apply_chunks_to_vol(chunks, vol, alpha, x0, y0, z0, n_devices, dim_chunk);
        }
      }
    }

    for(unsigned int device_idx = 0; device_idx < n_devices; device_idx++){
      cudaSetDevice(device_idx);
      cudaFree(dev_chunk[device_idx]);
      cudaFree(dev_proj_diff[device_idx]);
      cudaFree(dev_r_hats[device_idx]);
    }
    
  }

  free(data);
  free(threads);
  free(dev_chunk);
  free(dev_proj_diff);
  free(dev_r_hats);
  for(unsigned int device_idx = 0; device_idx < n_devices; device_idx++){
    free(chunks[device_idx]);
  }
  free(chunks);
  free(proj_idx_low);
  free(n_projs);
  for(unsigned int job_idx = 0; job_idx < n_jobs; job_idx++){
    free(r_hats[job_idx]);
    free(proj_diff_arrs[job_idx]);
  }
  free(proj_diff_arrs);
  free(r_hats);

}

