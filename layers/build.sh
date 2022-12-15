#!/bin/bash
cd ./build
rm -rf *
cmake ..
make
./layers_test