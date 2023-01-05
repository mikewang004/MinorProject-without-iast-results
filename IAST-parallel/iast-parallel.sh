#!/bin/bash

for p in */
do
        cd ${p}
        sbatch ./segiast_delftblue.sh
        cd ..
done

