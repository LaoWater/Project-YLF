

# easy cuda-compatibility install, #cuda-install, #pytorch-cuda
# Installing pytorch-cuda dependency (current version)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


import torch
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())
