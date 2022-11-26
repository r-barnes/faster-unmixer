#include "faster-unmixer.hpp"

#include <iostream>
#include <string>

int main(int argc, char **argv){
  if(argc != 3){
    std::cerr<<"Syntax: "<<argv[0]<<" <Flowdirs> <Sample Data"<<std::endl;
    return -1;
  }

  const std::string flowdirs_filename = argv[1];
  const std::string data_filename = argv[2];

  fastunmixer::faster_unmixer(flowdirs_filename, data_filename);

  return 0;
}