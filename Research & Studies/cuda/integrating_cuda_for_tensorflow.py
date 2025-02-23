# set TF_ENABLE_ONEDNN_OPTS=0 for tensorflow warning message

import os
import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print(tf.__version__)


# Tensorflow Cuda-CNN compatibility
# https://www.tensorflow.org/install/source#gpu
# Version	Python version	Compiler	Build tools	cuDNN	CUDA
# tensorflow-2.17.0	3.9-3.12	Clang 17.0.6	Bazel 6.5.0	8.9	12.3
# tensorflow-2.16.1	3.9-3.12	Clang 17.0.6	Bazel 6.5.0	8.9	12.3
# tensorflow-2.15.0	3.9-3.11	Clang 16.0.0	Bazel 6.1.0	8.9	12.2
# tensorflow-2.14.0	3.9-3.11	Clang 16.0.0	Bazel 6.1.0	8.7	11.8
# tensorflow-2.13.0	3.8-3.11	Clang 16.0.0	Bazel 5.3.0	8.6	11.8
# tensorflow-2.12.0	3.8-3.11	GCC 9.3.1	Bazel 5.3.0	8.6	11.8
# tensorflow-2.11.0	3.7-3.10	GCC 9.3.1	Bazel 5.3.0	8.1	11.2
# tensorflow-2.10.0	3.7-3.10	GCC 9.3.1	Bazel 5.1.1	8.1	11.2
