//#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include <stdio.h>
#include <stdlib.h>
#include "tools/macros.cuh"

/* struct OtherOpt { */
/*     OtherOpt() : deviceNumber(0), showHelp(false) {} */

/*     std::string processName; */
/*     int deviceNumber; */
/*     bool showHelp; */
    
/*     void setValidDeviceNumber( int i ) { */
/*         int deviceCount = 0; */
/*         cudaGetDeviceCount(&deviceCount); */
/*         if( i < 0 || i > deviceCount ) { */
/*             Catch::cout()<<"The device number is incorrect, please set valid cuda device number\n"; */
/*             exit(0); */
/*         } */
/*         deviceNumber = i; */

/*         cudaSetDevice(deviceNumber); */

/*         cudaDeviceProp deviceProp; */
/*         cudaGetDeviceProperties(&deviceProp, deviceNumber); */

/*         Catch::cout() <<"Device "<< deviceNumber <<": "<<deviceProp.name<<"\n"; */
/*     } */
/* }; */

void setValidDeviceNumber( int i ) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if( i < 0 || i > deviceCount ) {
        Catch::cout()<<"The device number is incorrect, please set valid cuda device number\n";
        exit(0);
    }
    int deviceNumber = i;

    cudaSetDevice(deviceNumber);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceNumber);

    if (if_debug()) {
        Catch::cout() <<"Device "<< deviceNumber <<": "<<deviceProp.name<<"\n";
    }
}

int main( int argc, char* const argv[] )
{
  Catch::Session session; // There must be exactly once instance

  // writing to session.configData() here sets defaults
  // this is the preferred way to set them

  int returnCode = session.applyCommandLine( argc, argv );
  if( returnCode != 0 ) // Indicates a command line error
    return returnCode;

  // writing to session.configData() or session.Config() here 
  // overrides command line args
  // only do this if you know you need to

    char* GPU_DEVICE;
    GPU_DEVICE = getenv ("GPU_DEVICE");
    int dev_id = 0;
    if (GPU_DEVICE != NULL)
        dev_id = atoi(GPU_DEVICE);

    if (dev_id >= 0) 
        setValidDeviceNumber(dev_id);

  return session.run();
}
