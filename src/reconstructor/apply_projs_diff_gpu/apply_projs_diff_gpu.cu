extern "C"{
  #include <reconstructor/apply_projs_diff_gpu/apply_projs_diff_gpu.h>
  #include <reconstructor/apply_projs_diff_gpu/apply_proj_to_chunk.h>
  
  #include <data/data_ops.h>
  #include <stdlib.h>
  #include <math.h>
}

void apply_chunk_to_vol(double* chunk, Data_3d* vol, double alpha,
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

void apply_projs_diff_gpu(Data_3d* vol, Data_3d* projs_diff, Data_2d* angles, 
  double alpha){

  unsigned int vdx = (vol->dim)[0];
  unsigned int vdy = (vol->dim)[1];
  unsigned int vdz = (vol->dim)[2];
  size_t free_mem, total_mem;
  cudaSetDevice(0);
  cudaMemGetInfo(&free_mem, &total_mem);
  unsigned int dim_chunk = (unsigned int)pow(free_mem*0.25/sizeof(double),1./3.);
  if(dim_chunk > vdx) dim_chunk = vdx;
  if(dim_chunk > vdy) dim_chunk = vdy;
  if(dim_chunk > vdz) dim_chunk = vdz;

  double* chunk = (double*)malloc(dim_chunk*dim_chunk*dim_chunk*sizeof(double));

  unsigned int n_proj = (projs_diff->dim)[0];
  unsigned int pdx = (projs_diff->dim)[1];
  unsigned int pdy = (projs_diff->dim)[2];
  double x_hat_i[3] = {1,0,0};
  double y_hat_i[3] = {0,1,0};
  double z_hat_i[3] = {0,0,1};
  double* r_hats = (double *)malloc(n_proj*3*3*sizeof(double));
  double v[3];
  for(unsigned int proj_i = 0; proj_i < n_proj; proj_i++){
    euler_rot(x_hat_i,(angles->data)[proj_i],v);
    r_hats[proj_i*9+0*3+0] = v[0];
    r_hats[proj_i*9+0*3+1] = v[1];
    r_hats[proj_i*9+0*3+2] = v[2];
    euler_rot(y_hat_i,(angles->data)[proj_i],v);
    r_hats[proj_i*9+1*3+0] = v[0];
    r_hats[proj_i*9+1*3+1] = v[1];
    r_hats[proj_i*9+1*3+2] = v[2];
    euler_rot(z_hat_i,(angles->data)[proj_i],v);
    r_hats[proj_i*9+2*3+0] = v[0];
    r_hats[proj_i*9+2*3+1] = v[1];
    r_hats[proj_i*9+2*3+2] = v[2];
  }
  double* proj_diff_arr = (double *)malloc(n_proj*pdy*pdx*sizeof(double));
  for(unsigned int p_idx = 0; p_idx < n_proj; p_idx++){
    for(unsigned int px = 0; px < pdx; px++){
      for(unsigned int py = 0; py < pdy; py++){
        proj_diff_arr[p_idx*pdy*pdx+py*pdx+px] = (projs_diff->data)[p_idx][px][py];
      }
    }
  }

  double *dev_chunk, *dev_proj_diff, *dev_r_hats;
  cudaMalloc( (void**)&dev_proj_diff, n_proj*pdx*pdy*sizeof(double) );
  cudaMalloc( (void**)&dev_r_hats, n_proj*3*3*sizeof(double) );
  cudaMalloc( (void**)&dev_chunk, dim_chunk*dim_chunk*dim_chunk*sizeof(double) );
  cudaMemcpy( dev_proj_diff, 
              proj_diff_arr,
              n_proj*pdx*pdy*sizeof(double),
              cudaMemcpyHostToDevice );
  cudaMemcpy( dev_r_hats, 
              r_hats,
              n_proj*3*3*sizeof(double),
              cudaMemcpyHostToDevice );

  for(unsigned int z0=0; z0<vdz; z0=z0+dim_chunk){
    for(unsigned int y0=0; y0<vdy; y0=y0+dim_chunk){
      for(unsigned int x0=0; x0<vdx; x0=x0+dim_chunk){
        apply_proj_to_chunk(dev_proj_diff, n_proj, pdx, pdy, dev_chunk,
          dim_chunk, x0, y0, z0, vdx, vdy, vdz, dev_r_hats);
        cudaMemcpy( chunk, 
                    dev_chunk,
                    dim_chunk*dim_chunk*dim_chunk*sizeof(double),
                    cudaMemcpyDeviceToHost );
        apply_chunk_to_vol(chunk, vol, alpha, x0, y0, z0, dim_chunk);
      }
    }
  }

  cudaFree( dev_chunk        );
  cudaFree( dev_proj_diff    );
  cudaFree( dev_r_hats       );
  free(chunk        );
  free(proj_diff_arr);
  free(r_hats       );

}

