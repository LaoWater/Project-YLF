import torch

x = torch.rand(5, 3)
print(x)

print(torch.backends.cudnn.version())

print(torch.cuda.is_available())
print(torch.version.cuda)  # This will tell you the CUDA version PyTorch is using.
print(torch.backends.cudnn.version())  # This will tell you the cuDNN version being used.

#
# Here's a summary of our situation:
#
# CUDA 12.6 is the global version (as shown by nvcc --version returning 12.6), meaning it's the default version available system-wide.
# CUDA 12.1 is installed and ready (as PyTorch is using this version), and it’s explicitly in the system path, ensuring that PyTorch can leverage it without any issues.
# cuDNN 9.5 supports CUDA 12.6, and while you don’t have a specific cuDNN for CUDA 12.1, cuDNN 9.5 is backward compatible with CUDA 12.1.
# Therefore, PyTorch can successfully work with CUDA 12.1 and cuDNN 9.5 together without issues.
# PyTorch confirms compatibility by returning torch.cuda.is_available() as True, meaning it's properly recognizing the CUDA 12.1 installation and using cuDNN as expected.
# In Conclusion:
# Everything fits together nicely:
#
# CUDA 12.1 is being used by PyTorch.
# CUDA 12.6 is globally installed, but PyTorch is sticking with 12.1.
# cuDNN 9.5 is backward compatible and works with CUDA 12.1, allowing the deep learning processes to run smoothly.
# Your setup is ready for running neural network computations on the GPU using PyTorch and Whisper! If you want, you can now proceed with testing CUDA-enabled PyTorch scripts for your projects.
