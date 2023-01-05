#!/bin/bash

cd fortran
gfortran -O3 -ffixed-line-length-256 -march=native -ffast-math *.f
./a.out
