CUDA_PATH	=	/usr/local/apps/cuda/cuda-10.1
CUDA_BIN_PATH	=	$(CUDA_PATH)/bin
CUDA_NVCC	=	$(CUDA_BIN_PATH)/nvcc

proj05:		proj05.cu
		$(CUDA_NVCC) -I/usr/local/apps/cuda/cuda-10.1/samples/common/inc/ -o proj05  proj05.cu

