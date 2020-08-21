#include <reconstructor/apply_projs_diff_st/apply_projs_diff_st.h>

#include <sys/sysinfo.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <math.h>
  
struct APST_Struct {
  Data_2d*     recon;
  Data_3d*     projs_diff;
  Data_2d*     angles;
  int*         wx;
  float*       ww2;
  float        alpha;
  unsigned int vdz;
  unsigned int idx_low;
  unsigned int idx_high;           
} APST_Struct;

void *exec_apply_proj(void *x_void_ptr){
  struct APST_Struct *x_ptr = (struct APST_Struct *)x_void_ptr;
  Data_2d*     recon      = x_ptr->recon;
  Data_3d*     projs_diff = x_ptr->projs_diff;
  Data_2d*     angles     = x_ptr->angles;
  int*         wx         = x_ptr->wx;
  float*       ww2        = x_ptr->ww2;
  float        alpha      = x_ptr->alpha;
  unsigned int vdz        = x_ptr->vdz;
  unsigned int idx_low    = x_ptr->idx_low;
  unsigned int idx_high   = x_ptr->idx_high;

  unsigned int vdxz    = (recon->dim)[1];
  unsigned int n_projs = (projs_diff->dim)[0];
  unsigned int pdx     = (projs_diff->dim)[2];
  int x;
  unsigned int y, xz;
  float w, f, recon_val;
  for(unsigned int tid = idx_low; tid < idx_high; tid++){
    y  = tid/vdxz;
    xz = tid - y*vdxz;
    recon_val = 0.;
    for(unsigned int pidx = 0; pidx < n_projs; pidx++){
      f = alpha*cos((angles->data)[pidx][1])/n_projs/vdz;
      x = wx[pidx*vdxz+xz];
      w = ww2[pidx*vdxz+xz];
      if((x>=0) & (x<(int)pdx)){
        recon_val = recon_val + f*w*(projs_diff->data)[pidx][y][x];
      }
      if((x+1>=0) & (x+1<(int)pdx)){
        recon_val = recon_val + f*(1.-w)*(projs_diff->data)[pidx][y][x+1];
      }
    }
    (recon->data)[y][xz] = (recon->data)[y][xz] + recon_val;
  }
  return 0;
}

void apply_projs_diff_st(Data_2d *recon, Data_3d *projs_diff, Data_2d *angles,
  int* wx, float* ww2, float alpha, unsigned int vdz){

  unsigned int num_cores = get_nprocs()-4;
  struct APST_Struct *data = (struct APST_Struct *)malloc(num_cores*sizeof(APST_Struct));
  pthread_t* threads = (pthread_t *)malloc(num_cores*sizeof(pthread_t));
  unsigned int vdy     = (recon->dim)[0];
  unsigned int vdxz    = (recon->dim)[1];
  unsigned int tid_inc = (vdy*vdxz)/num_cores+1;
  for(unsigned int core = 0; core < num_cores; core++){
    data[core].recon       = recon;
    data[core].projs_diff  = projs_diff;
    data[core].angles      = angles;
    data[core].wx          = wx;
    data[core].ww2         = ww2;
    data[core].alpha       = alpha;
    data[core].vdz         = vdz;
    data[core].idx_low     = core*tid_inc;
    data[core].idx_high    = (core+1)*tid_inc;           
    if(data[core].idx_high > vdy*vdxz){
      data[core].idx_high  = vdy*vdxz;           
    }
    pthread_create(&(threads[core]), NULL, exec_apply_proj, &(data[core]));
  }
  for(unsigned int core = 0; core < num_cores; core++){
    pthread_join(threads[core], NULL);
  }

}
