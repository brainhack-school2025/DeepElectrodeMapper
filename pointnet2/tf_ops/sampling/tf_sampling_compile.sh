#!/bin/bash

# TF2.13.1
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/targets/x86_64-linux/lib:/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH

rm -f tf_sampling_g.cu.o tf_sampling_so.so

# Pull in the same compile & link flags that TensorFlow uses
TF_CPPFLAGS=$(python3 - <<EOF
import tensorflow as tf
print(" ".join(tf.sysconfig.get_compile_flags()))
EOF
)
TF_LDFLAGS=$(python3 - <<EOF
import tensorflow as tf
print(" ".join(tf.sysconfig.get_link_flags()))
EOF
)

CUDA_DIR=/usr/local/cuda-12.3
CUDA_LIB_DIR=${CUDA_DIR}/lib64

# Compile the CUDA kernel
nvcc -std=c++17 -c tf_sampling_g.cu \
    ${TF_CPPFLAGS} \
    -I${CUDA_DIR}/include \
    -DGOOGLE_CUDA=1 \
    -x cu -Xcompiler -fPIC -O2 \
    -o tf_sampling_g.cu.o

# Compile and link everything into a shared object
g++ -std=c++17 -shared tf_sampling.cpp tf_sampling_g.cu.o \
    ${TF_CPPFLAGS} \
    -I${CUDA_DIR}/include \
    ${TF_LDFLAGS} \
    -L${CUDA_LIB_DIR} -lcudart \
    -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=1 \
    -o tf_sampling_so.so
