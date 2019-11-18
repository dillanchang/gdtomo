#include <reconstructor/calc_projs_diff.h>

#include <reconstructor/proj_calc.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <math.h>

typedef struct calc_proj_obj{
  Data_3d* projs_diff;
  Data_3d* vol;
  Data_3d* projs;
  Data_2d* angles;
  unsigned int N_idxs;
  unsigned int* idxs;
}calc_proj_obj;

void init_calc_proj_objs(calc_proj_obj* objs, Data_3d* projs_diff, Data_3d* vol, Data_3d*
  projs, Data_2d* angles, unsigned int num_cores){
  unsigned int s = (int)(ceil(1.*(projs->dim)[0]/num_cores));
  unsigned int low, high, n;
  for(unsigned int i=0; i<num_cores; i++){
    objs[i].projs_diff = projs_diff;
    objs[i].vol        = vol;
    objs[i].projs      = projs;
    objs[i].angles     = angles;
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

void *calc_projs_diff_helper(void *obj_ptr){
  calc_proj_obj* obj = (calc_proj_obj *)obj_ptr;
  Data_2d proj;
  unsigned int *proj_dim = (unsigned int *)malloc(2*sizeof(unsigned int));
  proj_dim[0] = ((obj->projs)->dim)[1];
  proj_dim[1] = ((obj->projs)->dim)[2];
  alloc_2d_data(&proj, proj_dim);
  unsigned int proj_idx;
  for(unsigned int idx=0; idx<(obj->N_idxs); idx++){
    proj_idx = (obj->idxs)[idx];
    calc_projection((obj->vol),((obj->angles)->data)[proj_idx],&proj);
    for(unsigned int x=0; x<((obj->projs)->dim)[1]; x++){
      for(unsigned int y=0; y<((obj->projs)->dim)[2]; y++){
        ((obj->projs_diff)->data)[proj_idx][x][y] =
          ((obj->projs)->data)[proj_idx][x][y] - (proj.data)[x][y];
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

void calc_projs_diff(Data_3d* projs_diff, Data_3d* vol, Data_3d* projs, Data_2d*
  angles, unsigned int num_cores){
  if(num_cores > (projs_diff->dim)[0]){
    num_cores = (projs_diff->dim)[0];
  }
  calc_proj_obj* objs = (calc_proj_obj *)malloc(num_cores*sizeof(calc_proj_obj));
  init_calc_proj_objs(objs, projs_diff, vol, projs, angles, num_cores);

  pthread_t *threads = (pthread_t *)malloc(num_cores*sizeof(pthread_t)); 
  for(unsigned int i=0; i<num_cores; i++){
    if(pthread_create(&(threads[i]), NULL, calc_projs_diff_helper,
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

  free_calc_proj_objs(objs,num_cores);
  free(objs);
}
