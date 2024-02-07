#!/bin/bash
echo "Number of Trials, Blocksize, MegaTrials/Second, Probability"
for t in 1024 4096 16384 65536 262144 1048576 2097152
do
        for b in 8 32 64 128
        do      
                /usr/local/apps/cuda/cuda-10.1/bin/nvcc -I/usr/local/apps/cuda/cuda-10.1/samples/common/inc/ -DNUMTRIALS=$t -DBLOCKSIZE=$b -o filename filename.cu
                ./filename
        done
done
