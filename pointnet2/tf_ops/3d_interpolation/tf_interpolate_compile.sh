#/bin/bash

# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF2.13.1
g++ -std=c++17 -shared -fPIC tf_interpolate.cpp -o tf_interpolate_so.so \
-I /home/jpdiaz/miniconda3/envs/pointnet2_env/lib/python3.8/site-packages/tensorflow/include \
-I /home/jpdiaz/miniconda3/envs/pointnet2_env/lib/python3.8/site-packages/tensorflow/include/external/nsync/public \
-O2 -D_GLIBCXX_USE_CXX11_ABI=1


