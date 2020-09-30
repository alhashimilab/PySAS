#!/bin/sh

g++  -I/usr/include/python2.7 \
     -I/usr/lib64/python2.7/site-packages/numpy/core/include \
     -fPIC -shared -lboost_python \
     -o pysasext.so pysasext.cpp -std=c++11
