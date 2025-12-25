#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终救命版：极小半径聚类提取
逻辑：利用车身点云密度远高于地台网格的特性，只提取最大的高密度集合。
"""

import os
import sys
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement

def final_rescue_clean(ply_path, output_path, eps=0.015, min_points=10):
    print(f"\n[CLEAN] 正在执行最终提取方案: {ply_path}")
    
    # 1. 读取原始数据
    plydata = PlyData.read(ply_path)
    v = plydata['vertex']
    xyz = np.stack([v['x'], v['y'], v['z']], axis=-1)
    
    # 2. 创建 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # 3. 统计滤波：先删掉那些明显的漂浮噪点
    print("  - 正在过滤背景噪点...")
    pcd, sor_indices = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
    
    # 4. 核心：极小半径聚类 (DBSCAN)
    # 我们把 eps 设得很小（1.5cm），这样地台那种稀疏的网格点就无法连在一起
    # 而车身这种紧密的点云会聚成一个巨大的核心
    print(f"  - 正在提取高密度车身 (eps={eps})...")
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    if labels.size == 0 or labels.max() < 0:
        print("[ERROR] 提取失败，请尝试增大 --eps 参数")
        return

    # 找到点数最多的那个簇（必定是车身）
    counts = np.bincount(labels[labels >= 0])
    largest_cluster_idx = np.argmax(counts)
    
    keep_indices_in_pcd = np.where(labels == largest_cluster_idx)[0]
    
    # 映射回原始索引
    final_keep_indices = sor_indices[keep_indices_in_pcd]
    
    # 5. 保存结果
    final_mask = np.zeros(len(xyz), dtype=bool)
    final_mask[final_keep_indices] = True
    new_vertices = v.data[final_mask]
    
    el = PlyElement.describe(new_vertices, 'vertex')
    PlyData([el]).write(output_path)
    
    print(f"[CLEAN] 提取完成！")
    print(f"  - 原始点数: {len(xyz)}")
    print(f"  - 保留点数: {len(new_vertices)} (车身)")
    print(f"  - 结果已保存至: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str)
    parser.add_argument("--eps", type=float, default=0.015, help="搜索半径，若车身被删则调大")
    args = parser.parse_args()
    
    output = args.output if args.output else args.input.replace(".ply", "_cleaned.ply")
    final_rescue_clean(args.input, output, eps=args.eps)
