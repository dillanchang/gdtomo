#include <reconstructor/reconstructor.h>

#include <reconstructor/calc_projs_st/calc_projs_st.h>
#include <reconstructor/calc_projs_diff.h>
#include <reconstructor/calc_projs_err.h>
#include <reconstructor/apply_projs_diff_st/apply_projs_diff_st.h>
#include <reconstructor/apply_positivity.h>
#include <data/data_ops.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void compute_weights(int** weights_xz, float** weights_w1, unsigned int* Nxz,
  int** weights_x, float** weights_w2, unsigned int vdx, unsigned int vdz,
  unsigned int pdx, Data_2d* angles){

  unsigned int n_projs = (angles->dim)[0];
  unsigned int vdxz    = vdx*vdz;

  *weights_x  =    (int *)malloc(n_projs*vdxz*sizeof(int));
  *weights_w2 =  (float *)malloc(n_projs*vdxz*sizeof(float));

  // Calculate r_hats
  float x_hat_i[3] = {1,0,0};
  float** r_hats = (float **)malloc(n_projs*sizeof(float *));
  float v[3];
  for(unsigned int i= 0; i < n_projs; i++){
    r_hats[i] = (float *)malloc(2*sizeof(float));
    euler_rot_rev(x_hat_i,(angles->data)[i],v);
    r_hats[i][0] = v[0];
    r_hats[i][1] = v[2];
  }

  // Calculate weights
  unsigned int idx;
  float x_p, z_p, x_p_rot, x_rot;
  int x_rot_int;
  unsigned int inc = 0;
  for(unsigned int x=0; x<vdx; x++){
    for(unsigned int z=0; z<vdz; z++){
      x_p = (float)((int)x-(int)(vdx/2));
      z_p = (float)((int)z-(int)(vdz/2));
      idx = vdz*x+z;
      for(unsigned int pidx = 0; pidx < n_projs; pidx++){
        x_p_rot = x_p*r_hats[pidx][0]
                + z_p*r_hats[pidx][1];
        x_rot = x_p_rot + 1.0*(int)(pdx/2);
        x_rot_int = (int)(floor(x_rot));
        
        if((x_rot_int >= 0) & (x_rot_int < (int)pdx)){
          inc = inc + 1;
        }
        if((x_rot_int+1 >= 0) & (x_rot_int+1 < (int)pdx)){
          inc = inc + 1;
        }
        (*weights_x)[pidx*vdxz+idx]  = x_rot_int;
        (*weights_w2)[pidx*vdxz+idx] = 1.-(x_rot-(float)x_rot_int);
      }
    }
  }

  unsigned int* x_count = (unsigned int *)malloc(n_projs*pdx*sizeof(unsigned int));
  for(unsigned int x = 0; x < n_projs*pdx; x++){
    x_count[x] = 0;
  }
  int x;
  for(unsigned int pidx = 0; pidx < n_projs; pidx++){
    for(unsigned int xz = 0; xz < vdxz; xz++){
      x = (*weights_x)[pidx*vdxz+xz];
      if(x >= 0 && x < (int)pdx){
        x_count[pidx*pdx+x] = x_count[pidx*pdx+x]+1;
      }
      if(x+1 >= 0 && x+1 < (int)pdx){
        x_count[pidx*pdx+x+1] = x_count[pidx*pdx+x+1]+1;
      }
    }
  }
  *Nxz = 0;
  for(unsigned int x = 0; x < n_projs*pdx; x++){
    if(x_count[x] > *Nxz){
      *Nxz = x_count[x];
    }
  }
  *weights_w1 = (float *)malloc(n_projs*pdx*(*Nxz)*sizeof(float));
  *weights_xz = (int *)malloc(n_projs*pdx*(*Nxz)*sizeof(int));
  for(unsigned int pidx = 0; pidx < n_projs; pidx++){
    for(unsigned int x=0; x<pdx; x++){
      for(unsigned int n=0; n<(*Nxz); n++){
        (*weights_xz)[pidx*pdx*(*Nxz)+x*(*Nxz)+n] = -1;
        (*weights_w1)[pidx*pdx*(*Nxz)+x*(*Nxz)+n] = 0.;
      }
    }
  }

  float w;
  for(unsigned int pidx = 0; pidx < n_projs; pidx++){
    for(unsigned int xz=0; xz<vdxz; xz++){
      x = (*weights_x)[pidx*vdxz+xz];
      w = (*weights_w2)[pidx*vdxz+xz];
      if(x >= 0 && x < (int)pdx){
        (*weights_xz)[pidx*pdx*(*Nxz)+x*(*Nxz)+x_count[pidx*pdx+x]-1] = xz;
        (*weights_w1)[pidx*pdx*(*Nxz)+x*(*Nxz)+x_count[pidx*pdx+x]-1] = w;
        x_count[pidx*pdx+x] = x_count[pidx*pdx+x] - 1;
      }
      if(x+1 >= 0 && x+1 < (int)pdx){
        (*weights_xz)[pidx*pdx*(*Nxz)+(x+1)*(*Nxz)+x_count[pidx*pdx+x+1]-1] = xz;
        (*weights_w1)[pidx*pdx*(*Nxz)+(x+1)*(*Nxz)+x_count[pidx*pdx+x+1]-1] = 1.-w;
        x_count[pidx*pdx+x+1] = x_count[pidx*vdx+x+1] - 1;
      }
    }
  }

  free(x_count);
  for(unsigned int i= 0; i < n_projs; i++){
    free(r_hats[i]);
  }
  free(r_hats);

}

void free_weights(int* weights_xz, float* weights_w1, int* weights_x, float* weights_w2){
  free(weights_xz);
  free(weights_w1);
  free(weights_x);
  free(weights_w2);
}

void calc_reconstruction_st(Data_3d* vol, Data_2d* angles, Data_3d* projs, Data_3d*
  err, Data_3d* projs_final, Recon_param* param){

  time_t start, end;
  int cpu_time_used;
  printf("%s\n", "Beginning reconstruction");
  start = time(NULL);

  unsigned int vdx     = (param->recon_dim)[0];
  unsigned int vdy     = (param->recon_dim)[1];
  unsigned int vdz     = (param->recon_dim)[2];
  unsigned int n_projs = (projs->dim)[0];
  unsigned int pdx     = (projs->dim)[1];
  unsigned int pdy     = (projs->dim)[2];

  Data_2d recon;
  unsigned int *recon_dim = (unsigned int *)malloc(2*sizeof(unsigned int));
  recon_dim[0] = vdy;
  recon_dim[1] = vdx*vdz;
  alloc_2d_data(&recon, recon_dim);
  for(unsigned int i=0; i < recon_dim[0]; i++){
    for(unsigned int j=0; j < recon_dim[1]; j++){
      (recon.data)[i][j] = 0.;
    }
  }

  Data_3d projs_curr;
  unsigned int *projs_curr_dim = (unsigned int *)malloc(3*sizeof(unsigned int));
  projs_curr_dim[0] = n_projs;
  projs_curr_dim[1] = pdy;
  projs_curr_dim[2] = pdx;    
  alloc_3d_data(&projs_curr, projs_curr_dim);

  Data_3d projs_diff;
  unsigned int *projs_diff_dim = (unsigned int *)malloc(3*sizeof(unsigned int));
  projs_diff_dim[0] = n_projs;
  projs_diff_dim[1] = pdy;    
  projs_diff_dim[2] = pdx;    
  alloc_3d_data(&projs_diff, projs_diff_dim);

  unsigned int *err_dim = (unsigned int *)malloc(3*sizeof(unsigned int));
  err_dim[0] = (param->n_iter);
  err_dim[1] = n_projs;
  err_dim[2] = 2;
  alloc_3d_data(err, err_dim);

  Data_2d err_iter;
  unsigned int *err_iter_dim = (unsigned int *)malloc(2*sizeof(unsigned int));
  err_iter_dim[0] = n_projs;
  err_iter_dim[1] = 2;
  alloc_2d_data(&err_iter, err_iter_dim);

  unsigned int Nxz;
  int    *weights_xz, *weights_x;
  float  *weights_w1, *weights_w2;
  compute_weights(&weights_xz, &weights_w1, &Nxz, &weights_x, &weights_w2, vdx, vdz, pdx, angles);

  float err_r1, err_r2;
  for(unsigned int iter=0; iter<(param->n_iter); iter++){
    calc_projs_st(&recon, &projs_curr, weights_xz, weights_w1, Nxz);
    calc_projs_diff_st(&projs_diff, projs, &projs_curr);
    calc_projs_err_st(&err_iter, &projs_diff, projs);
    apply_projs_diff_st(&recon, &projs_diff, angles, weights_x,
      weights_w2, (param->alpha), vdz);
    apply_positivity_st(&recon);
    err_r1 = 0; err_r2 = 0;
    for(unsigned int proj=0; proj<n_projs; proj++){
      (err->data)[iter][proj][0] = (err_iter.data)[proj][0];
      (err->data)[iter][proj][1] = (err_iter.data)[proj][1];
      err_r1 = err_r1 + (err_iter.data)[proj][0];
      err_r2 = err_r2 + (err_iter.data)[proj][1];
    }
    err_r1 = err_r1/n_projs;
    err_r2 = err_r2/n_projs;
    end = time(NULL);
    cpu_time_used = end-start;
    printf("%s%d%s%f%s%f%s%d%s\n", "Iteration ", iter+1, ":\tR1 Error: ",
      err_r1, ", R2 Error: ", err_r2, ", time elapsed: ", cpu_time_used, "s");
  }
  calc_projs_st(&recon, &projs_curr, weights_xz, weights_w1, Nxz);
  calc_projs_diff(&projs_diff, projs, &projs_curr);
  calc_projs_err(&err_iter, &projs_diff, projs);
  err_r1 = 0; err_r2 = 0;
  for(unsigned int proj=0; proj<n_projs; proj++){
    err_r1 = err_r1 + (err_iter.data)[proj][0];
    err_r2 = err_r2 + (err_iter.data)[proj][1];
  }
  err_r1 = err_r1/n_projs;
  err_r2 = err_r2/n_projs;
  end = time(NULL);
  cpu_time_used = end-start;
  printf("%s%f%s%f%s%d%s\n", "Final Result:\tR1 Error: ", err_r1,
    ", R2 Error: ", err_r2, ", time elapsed: ", cpu_time_used, "s");

  free_weights(weights_xz, weights_w1, weights_x, weights_w2);
  free_3d_data(&projs_diff);
  free_2d_data(&err_iter);

  unsigned int *vol_dim = (unsigned int *)malloc(3*sizeof(unsigned int));
  vol_dim[0] = vdx;
  vol_dim[1] = vdy;
  vol_dim[2] = vdz;
  alloc_3d_data(vol, vol_dim);
  for(unsigned int i=0; i<vdx; i++){
    for(unsigned int j=0; j<vdy; j++){
      for(unsigned int k=0; k<vdz; k++){
        (vol->data)[i][j][k] = (recon.data)[j][i*vdz+k];
      }
    }
  }
  free_2d_data(&recon);

  unsigned int *projs_final_dim = (unsigned int *)malloc(3*sizeof(unsigned int));
  projs_final_dim[0] = n_projs;
  projs_final_dim[1] = pdx;
  projs_final_dim[2] = pdy;
  alloc_3d_data(projs_final, projs_final_dim);
  for(unsigned int i=0; i<projs_final_dim[0]; i++){
    for(unsigned int j=0; j<projs_final_dim[1]; j++){
      for(unsigned int k=0; k<projs_final_dim[2]; k++){
        (projs_final->data)[i][j][k] = (projs_curr.data)[i][k][j];
      }
    }
  }
  free_3d_data(&projs_curr);
}
