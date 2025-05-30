#!/bin/bash

# Script to update TensorFlow C++ ops for compatibility with newer versions (TF 2.11+).
# In recent TensorFlow versions, Status::OK() has been deprecated in favor of tsl::OkStatus().
# This script recursively finds all .cpp files in the tf_ops directory and replaces
# all instances of Status::OK() with tsl::OkStatus().

# Root directory of pointnet++ repo
ROOT_DIR=tf_ops/

# Loop through all .cpp files in subdirectories
find "$ROOT_DIR" -type f -name "*.cpp" | while read -r cpp_file; do
    echo "Updating: $cpp_file"
    sed -i 's/\bStatus::OK()/tsl::OkStatus()/g' "$cpp_file"
done

echo "Done replacing Status::OK() with tsl::OkStatus()"
