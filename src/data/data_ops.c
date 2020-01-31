#include <data/data_ops.h>

#include <math.h>

void euler_rot(double* v, double* a, double* v_f){
  v_f[0] = 
    v[0]*cos(a[0])*cos(a[1]) + 
    v[2]*(cos(a[0])*cos(a[2])*sin(a[1]) + sin(a[0])*sin(a[2])) + 
    v[1]*(-cos(a[2])*sin(a[0]) + cos(a[0])*sin(a[1])*sin(a[2]));
  v_f[1] =
    v[0]*cos(a[1])*sin(a[0]) + 
    v[2]*(cos(a[2])*sin(a[0])*sin(a[1]) - cos(a[0])*sin(a[2])) + 
    v[1]*(cos(a[0])*cos(a[2]) + sin(a[0])*sin(a[1])*sin(a[2]));
  v_f[2] =
    v[2]*cos(a[1])*cos(a[2]) -
    v[0]*sin(a[1]) +
    v[1]*cos(a[1])*sin(a[2]);
}

void euler_rot_rev(double* v, double* a, double* v_f){
  v_f[0] = 
    v[0]*cos(a[0])*cos(a[1]) +
    v[1]*cos(a[1])*sin(a[0]) - 
    v[2]*sin(a[1]);
  v_f[1] =
    v[2]*cos(a[1])*sin(a[2]) + 
    v[0]*(-cos(a[2])*sin(a[0]) + cos(a[0])*sin(a[1])*sin(a[2])) + 
    v[1]*(cos(a[0])*cos(a[2]) + sin(a[0])*sin(a[1])*sin(a[2]));
  v_f[2] =
    v[2]*cos(a[1])*cos(a[2]) + 
    v[1]*(cos(a[2])*sin(a[0])*sin(a[1]) - cos(a[0])*sin(a[2])) + 
    v[0]*(cos(a[0])*cos(a[2])*sin(a[1]) + sin(a[0])*sin(a[2]));
}

void deg_to_rad(Data_2d* a){
  double f = PI/180.;
  for(unsigned int i=0; i<(a->dim)[0]; i++){
    for(unsigned int j=0; j<(a->dim)[1]; j++){
      (a->data)[i][j] = (a->data)[i][j]*f;
    }
  }
}

void rad_to_deg(Data_2d* a){
  double f = 180./PI;
  for(unsigned int i=0; i<(a->dim)[0]; i++){
    for(unsigned int j=0; j<(a->dim)[1]; j++){
      (a->data)[i][j] = (a->data)[i][j]*f;
    }
  }
}

int in_bounds(int i, int Ni){
  return (i>=0 && i<Ni);
}

