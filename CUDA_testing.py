#Run this to check if CUDA is working correctly
import torch
import sys
print(sys.version)
print(torch.cuda.is_available())

#check if ur GPU is CUDA compatible here: https://developer.nvidia.com/cuda-gpus
#Download CUDA here: https://developer.nvidia.com/cuda-downloads
