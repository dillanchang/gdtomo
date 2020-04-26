#include <reconstructor/apply_positivity.h>

void apply_positivity(Data_3d* recon){
  for(unsigned int i=0; i<(recon->dim)[0]; i++){
    for(unsigned int j=0; j<(recon->dim)[1]; j++){
      for(unsigned int k=0; k<(recon->dim)[2]; k++){
        if((recon->data)[i][j][k] < 0.){
          (recon->data)[i][j][k] = 0.;
        }
      }
    }
  }
}
