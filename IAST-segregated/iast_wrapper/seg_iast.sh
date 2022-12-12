#!/bin/bash

cd fortran
ftnchek -portability=all -nopretty -columns=131 *.f
gfortran -O3 -ffixed-line-length-256 -march=native -ffast-math *.f
./a.out
