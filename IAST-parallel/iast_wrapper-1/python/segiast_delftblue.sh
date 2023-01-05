
module load slurm
module load 2022r2
module load py-numpy 
module load py-scipy
module load py-matplotlib
srun -n 1 --mem-per-cpu=8GB --time=24:00:00 --partition=compute python3 autosegiast.py --job-name="autosegiast-delftblue-600K-5mols"
echo "Done!"