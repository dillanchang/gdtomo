#include <reconstructor/reconstructor.h>

#include <reconstructor/calc_projs.h>
#include <reconstructor/calc_projs_diff.h>
#include <reconstructor/calc_projs_err.h>
#include <reconstructor/apply_projs_diff.h>
#include <reconstructor/apply_positivity.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void calc_reconstruction(Data_3d* vol, Data_2d* angles, Data_3d* projs, Data_3d*
  err, Recon_param* param){

  time_t start, end;
  int cpu_time_used;
  printf("%s\n", "Beginning reconstruction");
  start = time(NULL);
  unsigned int *vol_dim = (unsigned int *)malloc(3*sizeof(unsigned int));
  vol_dim[0] = (param->recon_dim)[0];
  vol_dim[1] = (param->recon_dim)[1];
  vol_dim[2] = (param->recon_dim)[2];
  alloc_3d_data(vol, vol_dim);
  for(unsigned int i=0; i<vol_dim[0]; i++){
    for(unsigned int j=0; j<vol_dim[1]; j++){
      for(unsigned int k=0; k<vol_dim[2]; k++){
        (vol->data)[i][j][k] = 0.;
      }
    }
  }

  unsigned int *err_dim = (unsigned int *)malloc(3*sizeof(unsigned int));
  err_dim[0] = (param->n_iter);
  err_dim[1] = (projs->dim)[0];
  err_dim[2] = 2;
  alloc_3d_data(err, err_dim);

  Data_3d projs_curr;
  unsigned int *projs_curr_dim = (unsigned int *)malloc(3*sizeof(unsigned int));
  projs_curr_dim[0] = (projs->dim)[0];
  projs_curr_dim[1] = (projs->dim)[1];
  projs_curr_dim[2] = (projs->dim)[2];
  alloc_3d_data(&projs_curr, projs_curr_dim);

  Data_3d projs_diff;
  unsigned int *projs_diff_dim = (unsigned int *)malloc(3*sizeof(unsigned int));
  projs_diff_dim[0] = (projs->dim)[0];
  projs_diff_dim[1] = (projs->dim)[1];
  projs_diff_dim[2] = (projs->dim)[2];
  alloc_3d_data(&projs_diff, projs_diff_dim);

  Data_2d err_iter;
  unsigned int *err_iter_dim = (unsigned int *)malloc(2*sizeof(unsigned int));
  err_iter_dim[0] = (projs->dim)[0];
  err_iter_dim[1] = 2;
  alloc_2d_data(&err_iter, err_iter_dim);

  double err_r1, err_r2;
  for(unsigned int iter=0; iter<(param->n_iter); iter++){
    calc_projs(&projs_curr, vol, angles, param->num_cores);
    calc_projs_diff(&projs_diff, projs, &projs_curr);
    calc_projs_err(&err_iter, &projs_diff, projs);
    apply_projs_diff(vol, &projs_diff, angles, param->alpha, param->num_cores);
    apply_positivity(vol);
    err_r1 = 0; err_r2 = 0;
    for(unsigned int proj=0; proj<(projs->dim)[0]; proj++){
      (err->data)[iter][proj][0] = (err_iter.data)[proj][0];
      (err->data)[iter][proj][1] = (err_iter.data)[proj][1];
      err_r1 = err_r1 + (err_iter.data)[proj][0];
      err_r2 = err_r2 + (err_iter.data)[proj][1];
    }
    err_r1 = err_r1/(projs->dim)[0];
    err_r2 = err_r2/(projs->dim)[0];
    end = time(NULL);
    cpu_time_used = end-start;
    printf("%s%d%s%f%s%f%s%d%s\n", "Iteration ", iter+1, ":\tR1 Error: ",
      err_r1, ", R2 Error: ", err_r2, ", time elapsed: ", cpu_time_used, "s");
  }
  calc_projs(&projs_curr, vol, angles, param->num_cores);
  calc_projs_diff(&projs_diff, projs, &projs_curr);
  calc_projs_err(&err_iter, &projs_diff, projs);
  err_r1 = 0; err_r2 = 0;
  for(unsigned int proj=0; proj<(projs->dim)[0]; proj++){
    err_r1 = err_r1 + (err_iter.data)[proj][0];
    err_r2 = err_r2 + (err_iter.data)[proj][1];
  }
  err_r1 = err_r1/(projs->dim)[0];
  err_r2 = err_r2/(projs->dim)[0];
  end = time(NULL);
  cpu_time_used = end-start;
  printf("%s%f%s%f%s%d%s\n", "Final Result:\tR1 Error: ", err_r1,
    ", R2 Error: ", err_r2, ", time elapsed: ", cpu_time_used, "s");

  free_3d_data(&projs_curr);
  free_3d_data(&projs_diff);
  free_2d_data(&err_iter);
}
