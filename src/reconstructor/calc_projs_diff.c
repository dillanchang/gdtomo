#include <reconstructor/calc_projs_diff.h>

void calc_projs_diff(Data_3d* projs_diff, Data_3d* projs, Data_3d* projs_curr){
  for(unsigned int idx=0; idx<(projs->dim)[0]; idx++){
    for(unsigned int x=0; x<(projs->dim)[1]; x++){
      for(unsigned int y=0; y<(projs->dim)[2]; y++){
        (projs_diff->data)[idx][x][y] =
          (projs->data)[idx][x][y] - (projs_curr->data)[idx][x][y];
      }
    }
  }
}

void calc_projs_diff_st(Data_3d* projs_diff, Data_3d* projs, Data_3d* projs_curr){
  for(unsigned int idx=0; idx<(projs->dim)[0]; idx++){
    for(unsigned int x=0; x<(projs->dim)[1]; x++){
      for(unsigned int y=0; y<(projs->dim)[2]; y++){
        (projs_diff->data)[idx][y][x] =
          (projs->data)[idx][x][y] - (projs_curr->data)[idx][y][x];
      }
    }
  }
}
