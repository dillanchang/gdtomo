#include <stdio.h>
#include <string.h>
#include <sys/sysinfo.h>

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

  if(argc == 2 && (strcmp(argv[1],"test")==0)){
    return run_test();
  }
  if(argc == 2){
    const char * recon_info = argv[1];
    return run_gdtomo_recon(recon_info);
  }
  printf("%s\n", "Invalid Option");
  return 0;
}

