# parallel-programming-5

### Monte Carlo Simulation in CUDA
Re-implementation of the C (parallel-programming-1) Monte Carlo simulation using CUDA. This simulation determines the probability of a pin fitting into a hole. The underlying algorithm is the same as the first simulation (parallel programming-1) only designed to run on a GPU rather than a CPU. 

### Commentary

1. Machine/environment:
	- Machine used: `NVIDIA DGX`
	- Operating system used: `Linux`
	- Compiler driver used: `nvcc`
	
2. New probability estimate:
	- 26.9% vs the old estimate of 27.2%. The results are very close meaning both simulations are producing consistent results. This shows that both CPU and GPU computations (undergoing the same algorithm) are agnostic to the level of parallelism. This is probably a good thing and indicates that the performance increase while using the GPU still yields consistent results as both simulations converge around 27% success likelihood as trial size increases.
	
3. Performance numbers and graphs

	1. GPU Performance numbers
	![[Screenshot 2023-05-19 at 16.52.04.png]]
	2. Performance vs. Numtrials
	![[performance_numtrials.svg]]
	3. Performance vs. Blocksize
	![[performance_blocksize.svg]]
	
4. Patterns in the performance curves
	- Observation in blocksize scaling: as the blocksize increases the performance improves up to a certain point and then there are diminishing returns.
	- Observation in numtrials scaling: as the number of trials increases there is a higher trend in performance. 
5. Pattern observation reasons
	- Blocksize scaling leading to improved performance is most likely influenced negatively by GPU architecture bottlenecks and access to memory, hence, we see diminishing returns by increasing blocksize (32 appears to be a magic number here).
	- Numtrial scaling shows improved performance in the results. This is most likely a factor in approaching more accurate results by increasing sampling. As the number of trials is increased, the performance plateaus indicate a safe sample size. In the performance vs. blocksize chart, we could expect the higher sample size curves to begin to stack on top of each other. However, this observation probably depends on other variables like memory constraints and other factors because in some cases it appears that performance is scaling linearly with numtrials.
6. BLOCKSIZE of 8 is so much worse than the others
	- With a blocksize of 8, there are fewer threads executing concurrently. This might make the ratio of performance gains to overhead costs much greater in favor of overhead. There are likely many GPU cores not being used inside the block while waiting for all threads to synchronize within that block.
7. CPU performance results compared with the GPU run
	- This simulation using the GPU performed significantly better than the same simulation on a CPU. This is because there is more potential for significant speedup on a GPU which can accommodate much more parallelization of a problem depending on the computation ie better suited for operations like multiplication, not so much a complex control flow.
8. What does this mean for what you can do with GPU parallel computing?
	- GPU parallel computing can enable simulations, image/video-related tasks, matrix multiplication (ML training), and other applications to run much faster than CPU computing. It's important to note that not everything can be optimized or should be optimized to run on a GPU. In many cases, an algorithm might be better suited to be written for a CPU.
