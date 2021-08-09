#!/bin/sh 
#An example for MPI job. 
#SBATCH -J deduce
#SBATCH -o job-%j.log 
#SBATCH -e job-%j.err 
#SBATCH -N 1

echo Time is `date` 
echo Directory is $PWD 
echo This job runs on the following nodes: 
echo $SLURM_JOB_NODELIST 
echo This job has allocated $NPROCS cpu cores. 

srun python -u attribute_generator_sequential.py