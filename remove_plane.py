import numpy as np
import open3d as o3d
import argparse
import os
from plyfile import PlyData, PlyElement

def main():
    parser = argparse.ArgumentParser(description="Remove the ground plane from a PLY point cloud.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input PLY file.")
    parser.add_argument("--output", type=str, help="Path to the output PLY file.")
    parser.add_argument("--threshold", type=float, default=0.01, help="RANSAC distance threshold.")
    parser.add_argument("--y_limit", type=float, default=None, help="Optional Y coordinate limit. Points with Y > y_limit will be removed (assuming roof is -y, so +y is down).")
    parser.add_argument("--remove_below", action="store_true", help="If true, remove all points 'below' the plane (towards +y if roof is -y).")
    parser.add_argument("--num_planes", type=int, default=1, help="Number of planes to remove iteratively.")
    parser.add_argument("--remove_outliers", action="store_true", help="Use statistical outlier removal.")
    parser.add_argument("--keep_largest_cluster", action="store_true", help="Only keep the largest connected cluster (removes floating noise).")
    parser.add_argument("--cluster_eps", type=float, default=0.01, help="DBSCAN clustering epsilon.")
    parser.add_argument("--cluster_min_points", type=int, default=10, help="DBSCAN clustering min points.")
    parser.add_argument("--sor_nb_neighbors", type=int, default=20, help="SOR number of neighbors.")
    parser.add_argument("--sor_std_ratio", type=float, default=2.0, help="SOR standard deviation ratio.")
    parser.add_argument("--rotate_axis", type=str, choices=['x', 'y', 'z'], help="Axis to rotate around.")
    parser.add_argument("--rotate_angle", type=float, default=0, help="Angle to rotate in degrees.")
    parser.add_argument("--remove_all_planes", action="store_true", help="If true, iteratively remove planes until car is left.")
    
    args = parser.parse_args()
    
    if not args.output:
        base, ext = os.path.splitext(args.input)
        if base.endswith("_origin"):
            args.output = f"{base[:-7]}{ext}"
        else:
            args.output = f"{base}_no_plane{ext}"
    
    print(f"Reading {args.input}...")
    # Load with plyfile to preserve all properties (like color, normals)
    plydata = PlyData.read(args.input)
    v = plydata['vertex']
    
    # Extract points and other rotatable properties
    points = np.stack([v['x'], v['y'], v['z']], axis=-1)
    
    # Handle rotation if requested
    if args.rotate_axis and args.rotate_angle != 0:
        from scipy.spatial.transform import Rotation as R_scipy
        print(f"Rotating around {args.rotate_axis} by {args.rotate_angle} degrees...")
        angle_rad = np.radians(args.rotate_angle)
        
        # Create rotation matrix
        if args.rotate_axis == 'x':
            rot_matrix = R_scipy.from_euler('x', angle_rad).as_matrix()
        elif args.rotate_axis == 'y':
            rot_matrix = R_scipy.from_euler('y', angle_rad).as_matrix()
        else: # z
            rot_matrix = R_scipy.from_euler('z', angle_rad).as_matrix()
            
        # 1. Rotate points
        points = (rot_matrix @ points.T).T
        v.data['x'] = points[:, 0]
        v.data['y'] = points[:, 1]
        v.data['z'] = points[:, 2]
        
        # 2. Rotate normals if they exist
        if all(name in v.data.dtype.names for name in ['nx', 'ny', 'nz']):
            print("Rotating normals...")
            normals = np.stack([v['nx'], v['ny'], v['nz']], axis=-1)
            normals = (rot_matrix @ normals.T).T
            v.data['nx'] = normals[:, 0]
            v.data['ny'] = normals[:, 1]
            v.data['nz'] = normals[:, 2]
            
        # 3. Rotate Gaussian quaternions if they exist (rot_0, rot_1, rot_2, rot_3)
        # 3DGS convention: rot_0 is w, (rot_1, rot_2, rot_3) is (x, y, z)
        if all(name in v.data.dtype.names for name in ['rot_0', 'rot_1', 'rot_2', 'rot_3']):
            print("Rotating Gaussian orientations...")
            # Extract as (x, y, z, w) for Scipy
            qs = np.stack([v['rot_1'], v['rot_2'], v['rot_3'], v['rot_0']], axis=-1)
            
            # Normalize to be safe
            norm = np.linalg.norm(qs, axis=-1, keepdims=True)
            qs = qs / (norm + 1e-8)
            
            q_objs = R_scipy.from_quat(qs)
            r_obj = R_scipy.from_matrix(rot_matrix)
            
            # Apply scene rotation: R * Q
            new_q_objs = r_obj * q_objs
            new_qs = new_q_objs.as_quat() # returns (x, y, z, w)
            
            v.data['rot_0'] = new_qs[:, 3] # w
            v.data['rot_1'] = new_qs[:, 0] # x
            v.data['rot_2'] = new_qs[:, 1] # y
            v.data['rot_3'] = new_qs[:, 2] # z

    # Create Open3D point cloud for processing
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    num_points_orig = len(points)
    mask = np.ones(num_points_orig, dtype=bool)
    
    if args.y_limit is not None:
        print(f"Applying Y limit: Y < {args.y_limit}")
        mask = mask & (points[:, 1] < args.y_limit)
    else:
        # Create a working copy of the point cloud for iterative RANSAC
        current_pcd = o3d.geometry.PointCloud()
        current_pcd.points = o3d.utility.Vector3dVector(points)
        
        for i in range(args.num_planes):
            if len(current_pcd.points) < 3:
                break
                
            print(f"Finding dominant plane {i+1} using RANSAC...")
            plane_model, inliers = current_pcd.segment_plane(distance_threshold=args.threshold,
                                                             ransac_n=3,
                                                             num_iterations=1000)
            [a, b, c, d] = plane_model
            print(f"Plane {i+1} equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
            
            # Map inliers back to original indices
            # This is tricky because current_pcd changes. 
            # Better to re-calculate distance for all original points.
            dists = points[:, 0] * a + points[:, 1] * b + points[:, 2] * c + d
            
            if args.remove_below:
                # Ensure normal points 'down' (+y)
                if b < 0:
                    a, b, c, d = -a, -b, -c, -d
                    dists = -dists
                to_remove = dists > -args.threshold
                print(f"Removing points 'at or below' plane {i+1}")
            else:
                to_remove = np.abs(dists) < args.threshold
                print(f"Removing inliers of plane {i+1}")
                
            mask = mask & (~to_remove)
            
            # Update current_pcd for next iteration
            remaining_points = points[mask]
            if len(remaining_points) < 3:
                break
            current_pcd.points = o3d.utility.Vector3dVector(remaining_points)

    # Filter points
    filtered_v_data = v.data[mask]
    
    # Advanced cleaning
    if args.remove_outliers or args.keep_largest_cluster:
        # Create a new pcd from current masked points
        final_points = points[mask]
        pcd_final = o3d.geometry.PointCloud()
        pcd_final.points = o3d.utility.Vector3dVector(final_points)
        
        final_mask = np.ones(len(final_points), dtype=bool)
        
        if args.remove_outliers:
            print("Removing statistical outliers...")
            cl, ind = pcd_final.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            sor_mask = np.zeros(len(final_points), dtype=bool)
            sor_mask[ind] = True
            final_mask = final_mask & sor_mask
            # Update pcd for next step if needed
            pcd_final = pcd_final.select_by_index(ind)
        
        if args.keep_largest_cluster:
            print(f"Keeping only the largest cluster (eps={args.cluster_eps}, min_points={args.cluster_min_points})...")
            labels = np.array(pcd_final.cluster_dbscan(eps=args.cluster_eps, 
                                                       min_points=args.cluster_min_points, 
                                                       print_progress=False))
            if labels.max() >= 0:
                counts = np.bincount(labels[labels >= 0])
                print(f"Found {len(counts)} clusters. Top 5 sizes: {sorted(counts, reverse=True)[:5]}")
                largest_cluster_idx = np.argmax(counts)
                cluster_mask_in_pcd = (labels == largest_cluster_idx)
                
                # We need to map this back to the final_mask
                # Since we might have already removed points with SOR, 
                # pcd_final's indices correspond to points where final_mask was true
                temp_indices = np.where(final_mask)[0]
                cluster_indices = temp_indices[cluster_mask_in_pcd]
                
                new_final_mask = np.zeros(len(final_points), dtype=bool)
                new_final_mask[cluster_indices] = True
                final_mask = new_final_mask
            else:
                print("Warning: No clusters found!")

        filtered_v = filtered_v_data[final_mask]
    else:
        filtered_v = filtered_v_data
    
    print(f"Points before: {num_points_orig}")
    print(f"Points after: {len(filtered_v)}")
    print(f"Removed {num_points_orig - len(filtered_v)} points.")
    
    # Save output
    el = PlyElement.describe(filtered_v, 'vertex')
    PlyData([el]).write(args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()

