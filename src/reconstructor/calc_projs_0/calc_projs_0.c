#include <reconstructor/calc_projs.h>

#include <data/data_ops.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <math.h>

double vol_val(double x, double y, double z, Data_3d* vol){
  int Ni = (int)(vol->dim)[0];
  int Nj = (int)(vol->dim)[1];
  int Nk = (int)(vol->dim)[2];
  double x_0 = x-floor(x); int x_center = Ni/2; int i = (int)floor(x)+x_center;
  double y_0 = y-floor(y); int y_center = Nj/2; int j = (int)floor(y)+y_center;
  double z_0 = z-floor(z); int z_center = Nk/2; int k = (int)floor(z)+z_center;
  
  double q000, q100, q010, q001, q110, q101, q011, q111;
  if(in_bounds(i,Ni) && in_bounds(j,Nj) && in_bounds(k,Nk)){
    q000 = (vol->data)[i][j][k]; } else { q000 = 0.;
  }
  if(in_bounds(i+1,Ni) && in_bounds(j,Nj) && in_bounds(k,Nk)){
    q100 = (vol->data)[i+1][j][k]; } else { q100 = 0.;
  }
  if(in_bounds(i,Ni) && in_bounds(j+1,Nj) && in_bounds(k,Nk)){
    q010 = (vol->data)[i][j+1][k]; } else { q010 = 0.;
  }
  if(in_bounds(i,Ni) && in_bounds(j,Nj) && in_bounds(k+1,Nk)){
    q001 = (vol->data)[i][j][k+1]; } else { q001 = 0.;
  }
  if(in_bounds(i+1,Ni) && in_bounds(j+1,Nj) && in_bounds(k,Nk)){
    q110 = (vol->data)[i+1][j+1][k]; } else { q110 = 0.;
  }
  if(in_bounds(i+1,Ni) && in_bounds(j,Nj) && in_bounds(k+1,Nk)){
    q101 = (vol->data)[i+1][j][k+1]; } else { q101 = 0.;
  }
  if(in_bounds(i,Ni) && in_bounds(j+1,Nj) && in_bounds(k+1,Nk)){
    q011 = (vol->data)[i][j+1][k+1]; } else { q011 = 0.;
  }
  if(in_bounds(i+1,Ni) && in_bounds(j+1,Nj) && in_bounds(k+1,Nk)){
    q111 = (vol->data)[i+1][j+1][k+1]; } else { q111 = 0.;
  }
  return
    q000*(1-x_0)*(1-y_0)*(1-z_0) +
    q100*x_0*(1-y_0)*(1-z_0)     +
    q010*(1-x_0)*y_0*(1-z_0)     +
    q001*(1-x_0)*(1-y_0)*z_0     +
    q110*x_0*y_0*(1-z_0)         +
    q101*x_0*(1-y_0)*z_0         +
    q011*(1-x_0)*y_0*z_0         +
    q111*x_0*y_0*z_0;
}

void calc_one_projection(Data_3d* vol, double* angles, Data_2d* proj){
  int Nx_p    = (int)((proj->dim)[0]);
  int Ny_p    = (int)((proj->dim)[1]);
  int x_min   = -1*(int)((vol->dim)[0]/2);
  int y_min   = -1*(int)((vol->dim)[1]/2);
  int z_min   = -1*(int)((vol->dim)[2]/2);
  int z_max_p = (int)ceil(sqrt( (double)x_min*x_min
                              + (double)y_min*y_min
                              + (double)z_min*z_min ));
  int z_min_p = -1*z_max_p;

  double x_hat_i[3] = {1,0,0}; double x_hat[3];
  double y_hat_i[3] = {0,1,0}; double y_hat[3];
  double z_hat_i[3] = {0,0,1}; double z_hat[3];
  euler_rot_rev(x_hat_i,angles,x_hat);
  euler_rot_rev(y_hat_i,angles,y_hat);
  euler_rot_rev(z_hat_i,angles,z_hat);

  double x, y, z, x_p, y_p, z_p, v;
  for(int i_p = 0; i_p < Nx_p; i_p++){
    for(int j_p = 0; j_p < Ny_p; j_p++){
      x_p = (double)(i_p-Nx_p/2);
      y_p = (double)(j_p-Ny_p/2);
      (proj->data)[i_p][j_p] = 0.;
      for(int k_p = z_min_p; k_p <= z_max_p; k_p++){
        z_p = (double)k_p;
        x = x_p*x_hat[0]+y_p*x_hat[1]+z_p*x_hat[2];
        y = x_p*y_hat[0]+y_p*y_hat[1]+z_p*y_hat[2];
        z = x_p*z_hat[0]+y_p*z_hat[1]+z_p*z_hat[2];
        v = vol_val(x, y, z, vol);
        (proj->data)[i_p][j_p] = (proj->data)[i_p][j_p] + v;
      }
    }
  }
}

typedef struct calc_proj_obj{
  Data_3d* vol;
  Data_3d* projs;
  Data_2d* angles;
  unsigned int N_idxs;
  unsigned int* idxs;
}calc_proj_obj;

void init_calc_proj_objs(calc_proj_obj* objs, Data_3d* vol, Data_3d* projs,
  Data_2d* angles, unsigned int num_cores){
  unsigned int s = (int)(ceil(1.*(projs->dim)[0]/num_cores));
  unsigned int low, high, n;
  for(unsigned int i=0; i<num_cores; i++){
    objs[i].projs  = projs;
    objs[i].vol    = vol;
    objs[i].angles = angles;
    low = i*s;
    high = (i+1)*s-1;
    if(high > (projs->dim)[0]-1){
      high = (projs->dim)[0]-1;
    }
    n = high-low+1;
    objs[i].N_idxs     = n;
    objs[i].idxs       = (unsigned int *)malloc(n*sizeof(unsigned int));
    for(unsigned int j=0; j<n; j++){
      (objs[i].idxs)[j]  = low+j;
    }
  }
}

void* calc_proj_helper(void* obj_ptr){
  calc_proj_obj* obj = (calc_proj_obj *)obj_ptr;
  Data_2d proj;
  unsigned int *proj_dim = (unsigned int *)malloc(2*sizeof(unsigned int));
  proj_dim[0] = ((obj->projs)->dim)[1];
  proj_dim[1] = ((obj->projs)->dim)[2];
  alloc_2d_data(&proj, proj_dim);
  unsigned int k;
  for(unsigned int i=0; i<(obj->N_idxs); i++){
    k = (obj->idxs)[i];
    calc_one_projection(obj->vol,((obj->angles)->data)[k],&proj);
    for(unsigned int x=0; x<(proj_dim)[0]; x++){
      for(unsigned int y=0; y<(proj_dim)[1]; y++){
        ((obj->projs)->data)[k][x][y] = (proj.data)[x][y];
      }
    }
  }
  free_2d_data(&proj);
  return NULL;
}

void free_calc_proj_objs(calc_proj_obj* objs, unsigned int num_cores){
  for(unsigned int i=0; i<num_cores; i++){
    free(objs[i].idxs);
  }
}

void calc_projs_0(Data_3d* projs, Data_3d* vol, Data_2d* angles, unsigned int num_cores){
  if(num_cores > (projs->dim)[0]){
    num_cores = (projs->dim)[0];
  }
  calc_proj_obj* objs = (calc_proj_obj *)malloc(num_cores*sizeof(calc_proj_obj));
  init_calc_proj_objs(objs, vol, projs, angles, num_cores);
  pthread_t *threads = (pthread_t *)malloc(num_cores*sizeof(pthread_t)); 
  for(unsigned int i=0; i<num_cores; i++){
    if(pthread_create(&(threads[i]), NULL, calc_proj_helper,
      (void *)(&(objs[i])))){
      fprintf(stderr, "Error creating thread in calc_projs_diff.\n");
      exit(1);
    }
  }
  for(unsigned int i=0; i<num_cores; i++){
    if(pthread_join(threads[i], NULL)){
      fprintf(stderr, "Error joining thread in calc_projs_diff.\n");
      exit(1);
    }
  }
  free(threads);
  free_calc_proj_objs(objs,num_cores);
  free(objs);
}

