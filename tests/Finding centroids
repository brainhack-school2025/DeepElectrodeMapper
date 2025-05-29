import os
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# File paths
ply_path = "/Users/ivyzhong/Zhong_project/Data/processed_data/pointcloud_200k.ply"
npz_path = "/Users/ivyzhong/Zhong_project/Data/processed_data/pointcloud_clusters.npz"
target_clusters = 126

def analyze_data_for_dbscan(points, k=4):
    """Analyze data to find optimal DBSCAN parameters"""
    print("Analyzing data for optimal DBSCAN parameters...")
    
    # Calculate k-distance graph to find optimal eps
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(points)
    distances, indices = neighbors_fit.kneighbors(points)
    
    # Sort distances to k-th nearest neighbor
    distances = np.sort(distances[:, k-1], axis=0)
    
    # Find the "elbow" point (simple approach)
    # The optimal eps is usually where the curve has the steepest increase
    print(f"Distance statistics:")
    print(f"Min k-distance: {distances.min():.6f}")
    print(f"Max k-distance: {distances.max():.6f}")
    print(f"Mean k-distance: {distances.mean():.6f}")
    print(f"Median k-distance: {np.median(distances):.6f}")
    
    # Suggest eps range based on data
    suggested_eps_min = np.percentile(distances, 10)
    suggested_eps_max = np.percentile(distances, 90)
    print(f"Suggested eps range: {suggested_eps_min:.6f} to {suggested_eps_max:.6f}")
    
    return suggested_eps_min, suggested_eps_max

def try_dbscan_with_range(points, target_clusters, eps_min, eps_max, steps=50):
    """Try DBSCAN with a range of eps values"""
    print(f"\nTrying DBSCAN with eps from {eps_min:.6f} to {eps_max:.6f}...")
    
    eps_values = np.linspace(eps_min, eps_max, steps)
    results = []
    
    for eps in eps_values:
        clustering = DBSCAN(eps=eps, min_samples=5).fit(points)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        results.append({
            'eps': eps,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'labels': labels
        })
        
        if n_clusters == target_clusters:
            print(f"✓ Found exactly {target_clusters} clusters with eps={eps:.6f}")
            return labels, eps
        
        if len(results) % 10 == 0:  # Print progress every 10 iterations
            print(f"eps={eps:.6f}: {n_clusters} clusters, {n_noise} noise points")
    
    # Find closest result
    best_result = min(results, key=lambda x: abs(x['n_clusters'] - target_clusters))
    print(f"Best DBSCAN result: {best_result['n_clusters']} clusters with eps={best_result['eps']:.6f}")
    
    return best_result['labels'], best_result['eps']

if os.path.exists(npz_path):
    print("Loading from .npz file...")
    data = np.load(npz_path)
    points_down = data['points']
    labels = data['labels']
    centroids = data['centroids']
    
    # Check if we have the right number of clusters
    if len(centroids) != target_clusters:
        print(f"Loaded data has {len(centroids)} clusters, but we need {target_clusters}. Reclustering...")
        recalculate = True
    else:
        recalculate = False
        print(f"Loaded data has correct number of clusters: {len(centroids)}")
else:
    recalculate = True

if recalculate:
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"Loaded point cloud with {len(pcd.points)} points.")
    
    # Downsample
    voxel_size = 0.001
    pcd_down = pcd.voxel_down_sample(voxel_size)
    points_down = np.asarray(pcd_down.points)
    print(f"Downsampled to {len(points_down)} points.")
    
    # Option 1: Use K-means (guaranteed to give exact number of clusters)
    print(f"\n=== Using K-means for exactly {target_clusters} clusters ===")
    kmeans = KMeans(n_clusters=target_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(points_down)
    kmeans_centroids = kmeans.cluster_centers_
    
    # Option 2: Analyze and try DBSCAN (for comparison)
    print(f"\n=== Analyzing DBSCAN potential ===")
    eps_min, eps_max = analyze_data_for_dbscan(points_down)
    
    # Try DBSCAN with informed parameter range
    dbscan_labels, best_eps = try_dbscan_with_range(points_down, target_clusters, eps_min, eps_max)
    dbscan_unique_labels = np.unique(dbscan_labels[dbscan_labels >= 0])
    dbscan_centroids = np.array([points_down[dbscan_labels == label].mean(axis=0) 
                                for label in dbscan_unique_labels]) if len(dbscan_unique_labels) > 0 else np.array([])
    
    # Choose the best method
    dbscan_n_clusters = len(dbscan_centroids)
    kmeans_n_clusters = len(kmeans_centroids)
    
    print(f"\n=== Method Comparison ===")
    print(f"K-means: {kmeans_n_clusters} clusters (target: {target_clusters})")
    print(f"DBSCAN: {dbscan_n_clusters} clusters (target: {target_clusters})")
    
    if dbscan_n_clusters == target_clusters:
        print("Using DBSCAN results (exact match)")
        labels = dbscan_labels
        centroids = dbscan_centroids
        method_used = "DBSCAN"
    else:
        print("Using K-means results (guaranteed exact match)")
        labels = kmeans_labels
        centroids = kmeans_centroids
        method_used = "K-means"
    
    # Save results
    np.savez(npz_path, points=points_down, labels=labels, centroids=centroids, method=method_used)
    print(f"Saved clustering results to {npz_path}")

# Summary stats
n_clusters = len(centroids)
n_noise = list(labels).count(-1) if -1 in labels else 0
print(f"\n=== FINAL RESULTS ===")
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Number of centroids: {len(centroids)}")

# Verify we have exactly 125 centroids
if len(centroids) == target_clusters:
    print(f"✓ SUCCESS: Found exactly {target_clusters} centroids!")
else:
    print(f"⚠ WARNING: Found {len(centroids)} centroids instead of {target_clusters}")

# Print all centroids
print(f"\n=== ALL {len(centroids)} CENTROIDS ===")
for i, c in enumerate(centroids):
    print(f"Electrode {i+1:3d}: Centroid at [{c[0]:8.6f}, {c[1]:8.6f}, {c[2]:8.6f}]")

# Calculate some statistics about the centroids
if len(centroids) > 1:
    # Calculate distances between centroids
    from scipy.spatial.distance import pdist
    centroid_distances = pdist(centroids)
    print(f"\n=== CENTROID STATISTICS ===")
    print(f"Min distance between centroids: {centroid_distances.min():.6f}")
    print(f"Max distance between centroids: {centroid_distances.max():.6f}")
    print(f"Mean distance between centroids: {centroid_distances.mean():.6f}")

# Visualize: point cloud + centroids
print("\nPreparing visualization...")
pcd_down_vis = o3d.geometry.PointCloud()
pcd_down_vis.points = o3d.utility.Vector3dVector(points_down)

# Assign colors to clusters
max_label = labels.max() if labels.max() >= 0 else 0
if max_label > 0:
    # Use a colormap that can handle many clusters
    colors = plt.cm.tab20(labels % 20 / 20)  # Cycle through tab20 colormap
    colors[labels < 0] = [0, 0, 0, 1]  # Noise: black
else:
    colors = np.zeros((len(labels), 4))

pcd_down_vis.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Create centroid point cloud with larger spheres
centroid_spheres = []
for i, centroid in enumerate(centroids):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    sphere.translate(centroid)
    sphere.paint_uniform_color([1, 0, 0])  # Red
    centroid_spheres.append(sphere)

# Show visualization
print("Displaying visualization...")
o3d.visualization.draw_geometries([pcd_down_vis] + centroid_spheres)