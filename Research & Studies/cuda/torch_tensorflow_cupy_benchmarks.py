import torch
import tensorflow as tf
import cupy as cp
import time
import numpy as np


# Main function to benchmark all libraries
def testing_cuda_across_libs(matrix_size=10000, perform_additional_operation=True):
    def benchmark_torch():
        print("Running PyTorch...")

        # Torch CPU
        start_time = time.time()
        a_cpu = torch.randn(matrix_size, matrix_size)
        b_cpu = torch.randn(matrix_size, matrix_size)
        result_cpu = torch.mm(a_cpu, b_cpu)
        if perform_additional_operation:
            result_cpu = result_cpu * 0.5 + 2  # Element-wise weighting
        cpu_time = time.time() - start_time
        print(f"PyTorch CPU time: {cpu_time:.5f} seconds")

        # Torch GPU (if available)
        gpu_time = None
        if torch.cuda.is_available():
            start_time = time.time()
            a_gpu = a_cpu.cuda()
            b_gpu = b_cpu.cuda()
            result_gpu = torch.mm(a_gpu, b_gpu)
            if perform_additional_operation:
                result_gpu = result_gpu * 0.5 + 2  # Element-wise weighting
            gpu_time = time.time() - start_time
            print(f"PyTorch GPU time: {gpu_time:.5f} seconds")
        else:
            print("No CUDA-enabled GPU available for PyTorch.")

        return cpu_time, gpu_time

    def benchmark_tensorflow():
        print("Running TensorFlow...")

        # TensorFlow CPU
        start_time = time.time()
        a_cpu = tf.random.normal([matrix_size, matrix_size])
        b_cpu = tf.random.normal([matrix_size, matrix_size])
        result_cpu = tf.matmul(a_cpu, b_cpu)
        if perform_additional_operation:
            result_cpu = result_cpu * 0.5 + 2  # Element-wise weighting
        cpu_time = time.time() - start_time
        print(f"TensorFlow CPU time: {cpu_time:.5f} seconds")

        # TensorFlow GPU (if available)
        gpu_time = None
        if tf.config.list_physical_devices('GPU'):
            with tf.device('/GPU:0'):
                start_time = time.time()
                a_gpu = a_cpu
                b_gpu = b_cpu
                result_gpu = tf.matmul(a_gpu, b_gpu)
                if perform_additional_operation:
                    result_gpu = result_gpu * 0.5 + 2  # Element-wise weighting
                gpu_time = time.time() - start_time
                print(f"TensorFlow GPU time: {gpu_time:.5f} seconds")
        else:
            print("No CUDA-enabled GPU available for TensorFlow.")

        return cpu_time, gpu_time

    def benchmark_cupy():
        print("Running CuPy...")

        # CuPy CPU using NumPy (since CuPy doesn't run on CPU)
        start_time = time.time()
        a_cpu = np.random.randn(matrix_size, matrix_size)
        b_cpu = np.random.randn(matrix_size, matrix_size)
        result_cpu = np.dot(a_cpu, b_cpu)
        if perform_additional_operation:
            result_cpu = result_cpu * 0.5 + 2  # Element-wise weighting
        cpu_time = time.time() - start_time
        print(f"CuPy (NumPy) CPU time: {cpu_time:.5f} seconds")

        # CuPy GPU
        start_time = time.time()
        a_gpu = cp.random.randn(matrix_size, matrix_size)
        b_gpu = cp.random.randn(matrix_size, matrix_size)
        result_gpu = cp.dot(a_gpu, b_gpu)
        if perform_additional_operation:
            result_gpu = result_gpu * 0.5 + 2  # Element-wise weighting
        gpu_time = time.time() - start_time
        print(f"CuPy GPU time: {gpu_time:.5f} seconds")

        return cpu_time, gpu_time

    # Execute all benchmarks
    print("\n--- Matrix Multiplication Starting benchmarks ---\n")

    # PyTorch benchmark
    torch_cpu_time, torch_gpu_time = benchmark_torch()

    # TensorFlow benchmark
    tf_cpu_time, tf_gpu_time = benchmark_tensorflow()

    # CuPy benchmark
    cupy_cpu_time, cupy_gpu_time = benchmark_cupy()

    # Print the results
    print("\n--- Matrix Multiplication Benchmark Results ---")
    print(f"PyTorch CPU time: {torch_cpu_time:.5f} seconds")
    if torch_gpu_time is not None:
        print(f"PyTorch GPU time: {torch_gpu_time:.5f} seconds")
    else:
        print(f"PyTorch GPU: Not available")

    print(f"TensorFlow CPU time: {tf_cpu_time:.5f} seconds")
    if tf_gpu_time is not None:
        print(f"TensorFlow GPU time: {tf_gpu_time:.5f} seconds")
    else:
        print(f"TensorFlow GPU: Not available")

    print(f"CuPy (NumPy) CPU time: {cupy_cpu_time:.5f} seconds")
    print(f"CuPy GPU time: {cupy_gpu_time:.5f} seconds")


# Run the main function with user-defined parameters
if __name__ == "__main__":
    matrix_size = 10000  # You can change this to any size
    perform_additional_operation = False  # Change to False to skip element-wise weighting
    testing_cuda_across_libs(matrix_size, perform_additional_operation)
