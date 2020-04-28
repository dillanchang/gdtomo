#include <stdio.h>

extern "C" {
  #include <test/gpu.h>
}

extern "C"
void gpu_main()
{ 
  cudaDeviceProp prop;
  
  int count;
  cudaGetDeviceCount(&count);
  for (int i=0; i<count; i++){
    cudaGetDeviceProperties( &prop, i );
    printf("Name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Clock rate: %d\n", prop.clockRate);
    printf("Device copy overlap: %d\n", prop.deviceOverlap);
    printf("Total global mem: %ld\n", prop.totalGlobalMem);
    printf("Total shared mem per block: %ld\n", prop.sharedMemPerBlock);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }

}

