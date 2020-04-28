#include <reconstructor/calc_projs.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define DIM_CHUNK 15
#define PI 3.14159265

unsigned int get_num_chunks(Data_3d* vol){
  vol = vol;
  return 0;
}

void get_chunk(Data_3d* vol, unsigned int chunk_id, Data_3d* chunk,
  double* chunk_origin){
  vol = vol;
  chunk_id = chunk_id;
  chunk = chunk;
  chunk_origin = chunk_origin;
}

void get_query_pts(Data_3d* chunk, double* chunk_origin, Data_2d* angles,
  int** query_pts, unsigned int num_query_pts){
  chunk = chunk;
  chunk_origin = chunk_origin; 
  angles = angles;
  query_pts = query_pts;
  num_query_pts = num_query_pts;
}

void apply_query_pts(Data_3d* chunk, double* chunk_origin, Data_2d* angles,
  int** query_pts, Data_3d* projs){
  chunk = chunk;
  chunk_origin = chunk_origin; 
  angles = angles;
  query_pts = query_pts;
  projs = projs;
}

void clear_query_pts(int** query_pts, unsigned int num_query_pts){
  query_pts = query_pts;
  num_query_pts = num_query_pts;
}

void calc_projs_1(Data_3d* projs, Data_3d* vol, Data_2d* angles){

  Data_3d chunk;
  unsigned int num_chunks = get_num_chunks(vol);
  unsigned int *chunk_dim = (unsigned int *)malloc(3*sizeof(unsigned int));
  chunk_dim[0] = DIM_CHUNK;
  chunk_dim[1] = DIM_CHUNK;
  chunk_dim[2] = DIM_CHUNK;
  alloc_3d_data(&chunk, chunk_dim);

  double *chunk_origin = (double *)malloc(3*sizeof(double));
  double chunk_radius  = 1.73205*DIM_CHUNK/2.+5.;
  unsigned int num_query_pts = ceil(
                                (angles->dim)[0]*PI*chunk_radius*chunk_radius
                               );
  int **query_pts = (int **)malloc(num_query_pts*sizeof(int *));
  for(unsigned int pt_idx=0; pt_idx<num_query_pts; pt_idx++){
    query_pts[pt_idx] = (int *)malloc(5*sizeof(int));
  }
  clear_query_pts(query_pts, num_query_pts);

  for(unsigned int chunk_id=0; chunk_id<num_chunks; chunk_id++){
    get_chunk(vol, chunk_id, &chunk, chunk_origin);
    get_query_pts(&chunk, chunk_origin, angles, query_pts, num_query_pts);
    apply_query_pts(&chunk, chunk_origin, angles, query_pts, projs);
    clear_query_pts(query_pts, num_query_pts);
  }

  free_3d_data(&chunk);
  free(chunk_origin);
  for(unsigned int pt_idx=0; pt_idx<num_query_pts; pt_idx++){
    free(query_pts[pt_idx]);
  }
  free(query_pts);

}

