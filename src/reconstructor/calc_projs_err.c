#include <reconstructor/calc_projs_err.h>

#include <math.h>

void calc_projs_err(Data_2d* err, Data_3d* projs_diff, Data_3d* projs){
  float err_r1, err_r2, norm_err_r1, norm_err_r2;
  float a, b;
  for(unsigned int proj=0; proj<(projs->dim)[0]; proj++){
    err_r1 = 0.; err_r2 = 0.; norm_err_r1 = 0.; norm_err_r2 = 0.;
    for(unsigned int i=0; i<(projs->dim)[1]; i++){
      for(unsigned int j=0; j<(projs->dim)[2]; j++){
        err_r1      = err_r1 + fabs((projs_diff->data)[proj][i][j]);
        norm_err_r1 = norm_err_r1 + fabs((projs->data)[proj][i][j]);
        a = (projs_diff->data)[proj][i][j]*(projs_diff->data)[proj][i][j];
        b = (projs->data)[proj][i][j]*(projs->data)[proj][i][j];
        err_r2      = err_r2 + a*a;
        norm_err_r2 = norm_err_r2 + b*b;
      }
    }
    err_r1 = err_r1/norm_err_r1;
    err_r2 = sqrt(err_r2)/sqrt(norm_err_r2);
    (err->data)[proj][0] = err_r1;
    (err->data)[proj][1] = err_r2;
  }
}

void calc_projs_err_st(Data_2d* err, Data_3d* projs_diff, Data_3d* projs){
  float err_r1, err_r2, norm_err_r1, norm_err_r2;
  float a, b;
  for(unsigned int proj=0; proj<(projs->dim)[0]; proj++){
    err_r1 = 0.; err_r2 = 0.; norm_err_r1 = 0.; norm_err_r2 = 0.;
    for(unsigned int i=0; i<(projs->dim)[1]; i++){
      for(unsigned int j=0; j<(projs->dim)[2]; j++){
        err_r1      = err_r1 + fabs((projs_diff->data)[proj][j][i]);
        norm_err_r1 = norm_err_r1 + fabs((projs->data)[proj][i][j]);
        a = (projs_diff->data)[proj][j][i]*(projs_diff->data)[proj][j][i];
        b = (projs->data)[proj][i][j]*(projs->data)[proj][i][j];
        err_r2      = err_r2 + a*a;
        norm_err_r2 = norm_err_r2 + b*b;
      }
    }
    err_r1 = err_r1/norm_err_r1;
    err_r2 = sqrt(err_r2)/sqrt(norm_err_r2);
    (err->data)[proj][0] = err_r1;
    (err->data)[proj][1] = err_r2;
  }
}
