#include "faster-unmixer.hpp"

#include <iostream>
#include <string>

int main(int argc, char **argv){
  if(argc != 2){
    std::cerr<<"Syntax: "<<argv[0]<<" <Data dir>"<<std::endl;
    return -1;
  }

  const std::string data_dir = argv[1];

  fastunmixer::faster_unmixer(data_dir);

  return 0;
}