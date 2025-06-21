import h5py
import pandas as pd
from pyntcloud import PyntCloud

# Load and convert to DataFrame
cloud = PyntCloud.from_file("pointcloud_example.ply")
df = cloud.points

# Save to HDF5
with h5py.File("pointcloud_converted.h5", "w") as f:
    for col in df.columns:
        f.create_dataset(col, data=df[col].values)

# Visualize in notebook
cloud.plot()

import open3d as o3d

# Load point cloud (supports .ply, .pcd, .xyz, .xyzrgb, .txt, etc.)
pcd = o3d.io.read_point_cloud("pointcloud_example.ply")

# Print basic info
print(pcd)
print("Points:", len(pcd.points))

# Visualize
o3d.visualization.draw_geometries([pcd])