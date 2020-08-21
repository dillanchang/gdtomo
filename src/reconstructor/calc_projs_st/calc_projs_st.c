#include <reconstructor/calc_projs_st/calc_projs_st.h>

#include <sys/sysinfo.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>

struct PCST_Struct {
  Data_2d*     recon;
  Data_3d*     projs_curr;
  int*         wxz;
  float*       ww1;
  unsigned int Nxz;
  unsigned int idx_low;
  unsigned int idx_high;           
} PCST_Struct;

void *exec_calc_proj(void *x_void_ptr){
  struct PCST_Struct *x_ptr = (struct PCST_Struct *)x_void_ptr;
  Data_2d*     recon      = x_ptr->recon;
  Data_3d*     projs_curr = x_ptr->projs_curr;
  int*         wxz        = x_ptr->wxz;
  float*       ww1        = x_ptr->ww1;
  unsigned int Nxz        = x_ptr->Nxz;
  unsigned int idx_low    = x_ptr->idx_low;
  unsigned int idx_high   = x_ptr->idx_high;

  unsigned int pdy     = (projs_curr->dim)[1];
  unsigned int pdx     = (projs_curr->dim)[2];
  unsigned int pidx, y, x;
  int xz;
  float w, proj_val;
  for(unsigned int tid = idx_low; tid < idx_high; tid++){
    proj_val = 0.;
    pidx = tid/(pdy*pdx);    
    y    = (tid - pidx*pdy*pdx)/pdx;
    x    = tid - pidx*pdy*pdx - y*pdx;
    for(unsigned int n = 0; n < Nxz; n++){
      xz = wxz[pidx*pdx*Nxz+x*Nxz+n];
      if(xz < 0){
        break;
      }
      else{
        w  = ww1[pidx*pdx*Nxz+x*Nxz+n];
        proj_val = proj_val + w*(recon->data)[y][xz];
      }
    }
    (projs_curr->data)[pidx][y][x] = proj_val;
  }
  return 0;
}

void calc_projs_st(Data_2d *recon, Data_3d *projs_curr, int* wxz, float* ww1,
  unsigned int Nxz){
  unsigned int num_cores = get_nprocs()-4;
  struct PCST_Struct *data = (struct PCST_Struct *)malloc(num_cores*sizeof(PCST_Struct));
  pthread_t* threads = (pthread_t *)malloc(num_cores*sizeof(pthread_t));
  unsigned int N_projs = (projs_curr->dim)[0];
  unsigned int pdy     = (projs_curr->dim)[1];
  unsigned int pdx     = (projs_curr->dim)[2];
  unsigned int tid_inc = (N_projs*pdy*pdx)/num_cores+1;
  for(unsigned int core = 0; core < num_cores; core++){
    data[core].recon       = recon;
    data[core].projs_curr  = projs_curr;
    data[core].wxz         = wxz;
    data[core].ww1         = ww1;
    data[core].Nxz         = Nxz;
    data[core].idx_low     = core*tid_inc;
    data[core].idx_high    = (core+1)*tid_inc;           
    if(data[core].idx_high > N_projs*pdy*pdx){
      data[core].idx_high  = N_projs*pdy*pdx;           
    }
    pthread_create(&(threads[core]), NULL, exec_calc_proj, &(data[core]));
  }
  for(unsigned int core = 0; core < num_cores; core++){
    pthread_join(threads[core], NULL);
  }
}

