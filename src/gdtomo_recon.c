#include <data/data_types.h>
#include <data/file_io.h>
#include <data/data_ops.h>
#include <reconstructor/reconstructor.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mat.h"
#include "matrix.h"

void import_recon_data(const char* recon_data_fn, Data_3d* projs, Data_2d* angles,
  char** output_fn, Recon_param* param){

  MATFile* f;
  mxArray* a;
  double* data;
  f = matOpen(recon_data_fn, "r");

  // Read projections
  a = matGetVariable(f,"projs");
  data = (double *)mxGetPr(a);
  const long* projs_dim_raw = (const long *)mxGetDimensions(a);
  unsigned int *projs_dim = (unsigned int *)malloc(3*sizeof(unsigned int));
  projs_dim[0] = (unsigned int)(projs_dim_raw[0]);
  projs_dim[1] = (unsigned int)(projs_dim_raw[1]);
  projs_dim[2] = (unsigned int)(projs_dim_raw[2]);
  alloc_3d_data(projs, projs_dim);
  for(unsigned int i=0; i<projs_dim[0]; i++){
    for(unsigned int j=0; j<projs_dim[1]; j++){
      for(unsigned int k=0; k<projs_dim[2]; k++){
        (projs->data)[i][j][k] = data[i+projs_dim[0]*j+projs_dim[0]*projs_dim[1]*k];
      }
    }
  }
  mxDestroyArray(a);
  
  // Read angles
  a = matGetVariable(f,"angles");
  data = (double *)mxGetPr(a);
  const long* angles_dim_raw = (const long *)mxGetDimensions(a);
  unsigned int *angles_dim = (unsigned int *)malloc(2*sizeof(double));
  angles_dim[0] = (unsigned int )(angles_dim_raw[0]);
  angles_dim[1] = (unsigned int )(angles_dim_raw[1]);
  alloc_2d_data(angles, angles_dim);
  for(unsigned int i=0; i<angles_dim[0]; i++){
    for(unsigned int j=0; j<angles_dim[1]; j++){
      (angles->data)[i][j] = data[i+angles_dim[0]*j];
    }
  }
  mxDestroyArray(a);

  // Read output filename
  a = matGetVariable(f,"output_filename");
  unsigned short* output_fn_short = (unsigned short*)mxGetChars(a);
  unsigned int output_fn_len = 0;
  while(output_fn_short[output_fn_len] != 0){
    output_fn_len = output_fn_len + 1;
  }
  output_fn_len = output_fn_len + 1;
  *output_fn = (char *)malloc(output_fn_len*sizeof(char));
  for(unsigned int i = 0; i < output_fn_len-1; i++){
    (*output_fn)[i] = (char)(output_fn_short[i]);
  }
  (*output_fn)[output_fn_len-1] = 0;
  mxDestroyArray(a);

  a = matGetVariable(f,"n_iter");
  param->n_iter = (unsigned int)(mxGetPr(a)[0]);
  mxDestroyArray(a);

  a = matGetVariable(f,"alpha");
  param->alpha= mxGetPr(a)[0];
  mxDestroyArray(a);

  a = matGetVariable(f,"recon_dim");
  param->recon_dim = (unsigned int *)malloc(3*sizeof(double));
  (param->recon_dim)[0]= (unsigned int)((mxGetPr(a))[0]);
  (param->recon_dim)[1]= (unsigned int)((mxGetPr(a))[1]);
  (param->recon_dim)[2]= (unsigned int)((mxGetPr(a))[2]);
  mxDestroyArray(a);

  matClose(f);

}

void export_recon_data(Data_3d* recon, Data_3d* err, Data_3d* projs_final,
  char** output_fn){

  MATFile* f;
  f = matOpen((const char *)(*output_fn), "w");

  const size_t recon_dim[3] = {(recon->dim)[0],(recon->dim)[1],(recon->dim)[2]};
  mxArray *recon_mx = mxCreateNumericArray(3,recon_dim,mxDOUBLE_CLASS,0);
  double *recon_mx_ptr = (double *)mxGetPr(recon_mx);
  for(unsigned int i=0; i<recon_dim[0]; i++){
    for(unsigned int j=0; j<recon_dim[1]; j++){
      for(unsigned int k=0; k<recon_dim[2]; k++){
        recon_mx_ptr[i+recon_dim[0]*j+recon_dim[0]*recon_dim[1]*k] = (recon->data)[i][j][k];
      }
    }
  }
  matPutVariable(f,"recon",recon_mx);
  mxDestroyArray(recon_mx);

  const size_t err_dim[3] = {(err->dim)[0],(err->dim)[1],(err->dim)[2]};
  mxArray *err_mx = mxCreateNumericArray(3,err_dim,mxDOUBLE_CLASS,0);
  double *err_mx_ptr = (double *)mxGetPr(err_mx);
  for(unsigned int i=0; i<err_dim[0]; i++){
    for(unsigned int j=0; j<err_dim[1]; j++){
      for(unsigned int k=0; k<err_dim[2]; k++){
        err_mx_ptr[i+err_dim[0]*j+err_dim[0]*err_dim[1]*k] = (err->data)[i][j][k];
      }
    }
  }
  matPutVariable(f,"err",err_mx);
  mxDestroyArray(err_mx);

  const size_t projs_final_dim[3] = {(projs_final->dim)[0],(projs_final->dim)[1],(projs_final->dim)[2]};
  mxArray *projs_final_mx = mxCreateNumericArray(3,projs_final_dim,mxDOUBLE_CLASS,0);
  double *projs_final_mx_ptr = (double *)mxGetPr(projs_final_mx);
  for(unsigned int i=0; i<projs_final_dim[0]; i++){
    for(unsigned int j=0; j<projs_final_dim[1]; j++){
      for(unsigned int k=0; k<projs_final_dim[2]; k++){
        projs_final_mx_ptr[i+projs_final_dim[0]*j+projs_final_dim[0]*projs_final_dim[1]*k] = (projs_final->data)[i][j][k];
      }
    }
  }
  matPutVariable(f,"projs_final",projs_final_mx);
  mxDestroyArray(projs_final_mx);

  matClose(f);
}

int run_gdtomo_recon(const char* recon_data_fn){
  Data_3d projs;
  Data_2d angles;
  char* output_fn;
  Recon_param param;

  import_recon_data(recon_data_fn, &projs, &angles, &output_fn, &param);
  printf("%s\n", "Data import complete");

  deg_to_rad(&angles);

  Data_3d recon;
  Data_3d err;
  Data_3d projs_final;
  calc_reconstruction(&recon, &angles, &projs, &err, &projs_final, &param);

  export_recon_data(&recon, &err, &projs_final, &output_fn);

  printf("%s\n", "Data export complete");

  free_3d_data(&projs);
  free_2d_data(&angles);
  free_3d_data(&recon);
  free_3d_data(&err);
  free_3d_data(&projs_final);

  return 0;
}

