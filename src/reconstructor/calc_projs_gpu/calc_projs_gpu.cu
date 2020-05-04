extern "C"{
  #include <reconstructor/calc_projs_gpu/calc_projs_gpu.h>
  #include <reconstructor/calc_projs_gpu/apply_chunk_to_proj.h>
  #include <reconstructor/calc_projs_gpu/get_chunk_overlap.h>
  
  #include <data/data_ops.h>
  #include <stdlib.h>
  #include <math.h>
}

void calc_projs_gpu(Data_3d* projs, Data_3d* vol, Data_2d* angles){
  size_t free_mem, total_mem;
  cudaSetDevice(0);
  cudaMemGetInfo(&free_mem, &total_mem);
  unsigned int dim_chunk = (unsigned int)pow(1.0*(free_mem*0.25)/sizeof(double),1./3.);
  if(dim_chunk > (vol->dim)[0]+4) dim_chunk = (vol->dim)[0]+4;
  if(dim_chunk > (vol->dim)[1]+4) dim_chunk = (vol->dim)[1]+4;
  if(dim_chunk > (vol->dim)[2]+4) dim_chunk = (vol->dim)[2]+4;

  double* chunk = (double*)malloc(dim_chunk*dim_chunk*dim_chunk*sizeof(double));
  double* chunk_origin = (double *)malloc(3*sizeof(double));

  unsigned int n_proj = (projs->dim)[0];
  unsigned int pdx = (projs->dim)[1];
  unsigned int pdy = (projs->dim)[2];
  double x_hat_i[3] = {1,0,0};
  double y_hat_i[3] = {0,1,0};
  double z_hat_i[3] = {0,0,1};
  double* r_hats = (double *)malloc(n_proj*3*3*sizeof(double));
  double v[3];
  for(unsigned int proj_i = 0; proj_i < n_proj; proj_i++){
    euler_rot_rev(x_hat_i,(angles->data)[proj_i],v);
    r_hats[proj_i*9+0*3+0] = v[0];
    r_hats[proj_i*9+0*3+1] = v[1];
    r_hats[proj_i*9+0*3+2] = v[2];
    euler_rot_rev(y_hat_i,(angles->data)[proj_i],v);
    r_hats[proj_i*9+1*3+0] = v[0];
    r_hats[proj_i*9+1*3+1] = v[1];
    r_hats[proj_i*9+1*3+2] = v[2];
    euler_rot_rev(z_hat_i,(angles->data)[proj_i],v);
    r_hats[proj_i*9+2*3+0] = v[0];
    r_hats[proj_i*9+2*3+1] = v[1];
    r_hats[proj_i*9+2*3+2] = v[2];
  }
  double* proj_arr = (double *)malloc(n_proj*pdy*pdx*sizeof(double));
  for(unsigned int idx=0; idx<n_proj*pdy*pdx; idx++){
    proj_arr[idx] = 0.;
  }

  double x_min  = 1.0*(vol->dim)[0]/2;
  double y_min  = 1.0*(vol->dim)[1]/2;
  double z_min  = 1.0*(vol->dim)[2]/2;
  unsigned int lim_proj_z =
    (unsigned int) ceil(sqrt( (double)x_min*x_min
                            + (double)y_min*y_min
                            + (double)z_min*z_min ));

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
  int finished = get_chunk_overlap(vol, chunk_idx, chunk, dim_chunk, chunk_origin);
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
    finished = get_chunk_overlap(vol, chunk_idx, chunk, dim_chunk, chunk_origin);
  }

  cudaMemcpy( proj_arr, 
              dev_projs,
              n_proj*pdx*pdy*sizeof(double),
              cudaMemcpyDeviceToHost );
  for(unsigned int p_idx = 0; p_idx <n_proj; p_idx++){
    for(unsigned int py = 0; py < pdy; py++){
      for(unsigned int px = 0; px < pdx; px++){
        (projs->data)[p_idx][px][py] =
          proj_arr[p_idx*pdy*pdx+py*pdx+px];
      }
    }
  }

  cudaFree( dev_chunk        );
  cudaFree( dev_chunk_origin );
  cudaFree( dev_projs        );
  cudaFree( dev_r_hats       );
  free(chunk        );
  free(chunk_origin );
  free(proj_arr     );
  free(r_hats       );

}

