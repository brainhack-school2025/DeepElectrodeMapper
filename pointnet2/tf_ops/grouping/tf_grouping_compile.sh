#/bin/bash

#/usr/local/cuda-8.0/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF2.13.1
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/targets/x86_64-linux/lib:/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH

# Paths (adjust if needed)
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CUDA_DIR=/usr/local/cuda-12.3

# Step 1: Compile the CUDA code
$CUDA_DIR/bin/nvcc -std=c++17 -c -o tf_grouping_g.cu.o tf_grouping_g.cu \
    -I$TF_INC -I$TF_INC/external/nsync/public \
    -I$CUDA_DIR/include -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

# Step 2: Compile the shared object (no -ltensorflow_framework for TF2.13+)
g++ -std=c++17 -shared -o tf_grouping_so.so tf_grouping.cpp tf_grouping_g.cu.o \
    -I"$TF_INC" -I"$TF_INC/external/nsync/public" \
    -I"$CUDA_DIR/include" \
    -L"$CUDA_DIR/lib64" -lcudart \
    -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI=1
