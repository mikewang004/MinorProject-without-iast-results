#!/bin/bash

for q in */
do
   cd ${q}
    for p in */
    do
        cd ${p}
        sbatch ./segiast_delftblue.sh
        cd ..
done
cd ..
done
