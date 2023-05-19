#!/bin/bash
#SBATCH -J ZachTest
#SBATCH -A cs475-575
#SBATCH -p classgpufinal
#SBATCH --constraint=v100
#SBATCH -o proj05.out
#SBATCH -e proj05.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bochansz@oregonstate.edu

echo "Number of Trials, Blocksize, MegaTrials/Second, Probability"
for t in 1024 4096 16384 65536 262144 1048576 2097152
do
        for b in 8 32 64 128
        do      
                /usr/local/apps/cuda/cuda-10.1/bin/nvcc -I/usr/local/apps/cuda/cuda-10.1/samples/common/inc/ -DNUMTRIALS=$t -DBLOCKSIZE=$b -o proj05  proj05.cu
                ./proj05
        done
done
