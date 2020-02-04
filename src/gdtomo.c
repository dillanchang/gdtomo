#include <stdio.h>
#include <string.h>

#include <test/test.h>
#include <gdtomo_recon.h>
#include <gdtomo_calc_projs.h>

int main(int argc, char** argv)
{
  printf("%s\n", "#===========================================================================#");
  printf("%s\n", "#                                  gdtomo                                   #");
  printf("%s\n", "#                                                                           #");
  printf("%s\n", "#                                            github.com/dillanchang/gdtomo  #");
  printf("%s\n", "#===========================================================================#");

  if(argc == 3 && (strcmp(argv[1],"recon")==0)){
    const char * recon_info = argv[2];
    return run_gdtomo_recon(recon_info);
  }
  else if(argc == 3 && (strcmp(argv[1],"calc_projs")==0)){
    const char * calc_projs_info = argv[2];
    return run_gdtomo_calc_projs(calc_projs_info);
  }
  else if(argc == 2 && (strcmp(argv[1],"test")==0)){
    return run_test();
  }
  printf("%s\n", "Invalid Option");
  return 0;
}

