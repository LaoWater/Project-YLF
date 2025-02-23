import torch
import time


# Function to perform matrix multiplication and measure time
def matrix_multiplication(device):
    print(f"\nRunning on: {device}")

    # Define two large tensors (5000x5000 matrix)
    tensor1 = torch.rand((30000, 30000), device=device)
    tensor2 = torch.rand((30000, 30000), device=device)

    # Start the timer
    start_time = time.time()

    # Perform matrix multiplication
    result = torch.matmul(tensor1, tensor2)

    # Stop the timer
    end_time = time.time()

    print(f"Time taken on {device}: {end_time - start_time:.4f} seconds")


# Run on CPU
matrix_multiplication('cpu')

# Check if CUDA is available and run on GPU
if torch.cuda.is_available():
    matrix_multiplication('cuda')
else:
    print("CUDA is not available on this system.")
