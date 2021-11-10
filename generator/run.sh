#!/bin/sh 
#SBATCH -J generatedata
#SBATCH -o job-%j.log 
#SBATCH -e job-%j.err 
#SBATCH -p cpu

echo Time is `date` 
echo Directory is $PWD 
echo This job runs on the following nodes: 
echo $SLURM_JOB_NODELIST 
echo This job has allocated $NPROCS cores. 

#srun python -u attribute_generator_parallel.py
#srun python -u monotonicity_pair_generator.py
#srun python -u statistic.py
srun python -u run.py