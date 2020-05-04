#include <reconstructor/apply_projs_diff_cpu/apply_projs_diff_cpu.h>

#include <data/data_ops.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <math.h>

typedef struct apply_proj_obj{
  Data_3d* vol;
  Data_3d* projs_diff;
  Data_2d* angles;
  double   alpha;
  unsigned int N_idxs;
  unsigned int* idxs;
}apply_proj_obj;

double proj_interp_val(double** proj_diff, unsigned int Ni, unsigned int Nj,
  double x, double y){
  double x_0 = x-floor(x);     double y_0 = y-floor(y);
  int x_center = Ni/2;         int y_center = Nj/2;
  int i = floor(x)+x_center;   int j = floor(y)+y_center;
  double q11, q12, q21, q22;
  if(in_bounds(i,Ni) && in_bounds(j,Nj)){
    q11 = proj_diff[i][j]; } else{ q11 = 0.;
  }
  if(in_bounds(i+1,Ni) && in_bounds(j,Nj)){
    q21 = proj_diff[i+1][j]; } else{ q21 = 0.;
  }
  if(in_bounds(i,Ni) && in_bounds(j+1,Nj)){
    q12 = proj_diff[i][j+1]; } else{ q12 = 0.;
  }
  if(in_bounds(i+1,Ni) && in_bounds(j+1,Nj)){
    q22 = proj_diff[i+1][j+1]; } else{ q22 = 0.;
  }
  double r1 = (1-x_0)*q11+x_0*q21;
  double r2 = (1-x_0)*q12+x_0*q22;
  return (1-y_0)*r1+y_0*r2;
}

void update_vol_layer(Data_3d* vol, Data_3d* projs_diff, Data_2d* angles,
  unsigned int proj_idx, unsigned int k, unsigned int N_proj, double alpha){
  unsigned Ni= (vol->dim)[0]; int center_x = Ni/2;
  unsigned Nj= (vol->dim)[1]; int center_y = Nj/2;
  unsigned Nk= (vol->dim)[2]; int center_z = Nk/2;
  int x, y;
  int z = k-center_z;
  double rx, ry, v;
  double x_hat_i[3] = {1,0,0}; double x_hat[3];
  double y_hat_i[3] = {0,1,0}; double y_hat[3];
  double z_hat_i[3] = {0,0,1}; double z_hat[3];
  euler_rot(x_hat_i,(angles->data)[proj_idx],x_hat);
  euler_rot(y_hat_i,(angles->data)[proj_idx],y_hat);
  euler_rot(z_hat_i,(angles->data)[proj_idx],z_hat);
  for(unsigned int i=0; i<Ni; i++){
    for(unsigned int j=0; j<Nj; j++){
      x = i-center_x;
      y = j-center_y;
      rx = (double)x*x_hat[0]+(double)y*x_hat[1]+(double)z*x_hat[2];
      ry = (double)x*y_hat[0]+(double)y*y_hat[1]+(double)z*y_hat[2];
      v = proj_interp_val((projs_diff->data)[proj_idx],(projs_diff->dim)[1],
        (projs_diff->dim)[2],rx,ry);
      (vol->data)[i][j][k] = (vol->data)[i][j][k]+v*alpha*z_hat[2]/Nk/N_proj;
    }
  }
}

void init_apply_proj_objs(apply_proj_obj* objs, Data_3d* projs_diff, Data_3d* vol, Data_2d*
  angles, double alpha, unsigned int num_cores){
  unsigned int s = (int)(ceil(1.*(vol->dim)[2]/num_cores));
  unsigned int low, high, n;
  for(unsigned int i=0; i<num_cores; i++){
    objs[i].projs_diff = projs_diff;
    objs[i].vol        = vol;
    objs[i].angles     = angles;
    objs[i].alpha      = alpha;
    low = i*s;
    high = (i+1)*s-1;
    if(high > (vol->dim)[2]-1){
      high = (vol->dim)[2]-1;
    }
    n = high-low+1;
    objs[i].N_idxs     = n;
    objs[i].idxs       = (unsigned int *)malloc(n*sizeof(unsigned int));
    for(unsigned int j=0; j<n; j++){
      (objs[i].idxs)[j]  = low+j;
    }
  }
}

void* apply_projs_diff_helper(void* obj_ptr){
  apply_proj_obj* obj = (apply_proj_obj *)obj_ptr;
  unsigned int k;
  for(unsigned int i=0; i<(obj->N_idxs); i++){
    k = (obj->idxs)[i];
    for(unsigned int proj_idx=0; proj_idx<((obj->projs_diff)->dim)[0]; proj_idx++){
      update_vol_layer(obj->vol, obj->projs_diff, obj->angles, proj_idx, k,
        ((obj->projs_diff)->dim)[0], obj->alpha);
    }
  }
  return NULL;
}

void free_apply_proj_objs(apply_proj_obj* objs, unsigned int num_cores){
  for(unsigned int i=0; i<num_cores; i++){
    free(objs[i].idxs);
  }
}

void apply_projs_diff_cpu(Data_3d* vol, Data_3d* projs_diff, Data_2d* angles, 
  double alpha, unsigned int num_cores){
  if(num_cores > (vol->dim)[2]){
    num_cores = (vol->dim)[2];
  }
  apply_proj_obj* objs = (apply_proj_obj *)malloc(num_cores*sizeof(apply_proj_obj));
  init_apply_proj_objs(objs, projs_diff, vol, angles, alpha, num_cores);
  pthread_t *threads = (pthread_t *)malloc(num_cores*sizeof(pthread_t)); 
  for(unsigned int i=0; i<num_cores; i++){
    if(pthread_create(&(threads[i]), NULL, apply_projs_diff_helper,
      (void *)(&(objs[i])))){
      fprintf(stderr, "Error creating thread in apply_projs_diff.\n");
      exit(1);
    }
  }
  for(unsigned int i=0; i<num_cores; i++){
    if(pthread_join(threads[i], NULL)){
      fprintf(stderr, "Error joining thread in apply_projs_diff.\n");
      exit(1);
    }
  }
  free(threads);
  free_apply_proj_objs(objs,num_cores);
  free(objs);
}
