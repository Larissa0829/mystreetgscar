#!/usr/bin/env python3
"""
调试 sample 对象的颜色信息
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(__file__))

from lib.config import cfg
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.models.street_gaussian_model import StreetGaussianModel

def check_sample_colors():
    print("=" * 80)
    print("检查 Sample 对象的颜色信息")
    print("=" * 80)
    
    # 加载模型
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)
    
    print(f"\n已加载模型，迭代: {scene.loaded_iter}")
    print(f"对象列表: {gaussians.obj_list}")
    
    # 检查每个 sample 对象
    for obj_name in gaussians.obj_list:
        if not obj_name.endswith('_sample'):
            continue
        
        print(f"\n{'='*60}")
        print(f"检查: {obj_name}")
        print(f"{'='*60}")
        
        actor = getattr(gaussians, obj_name)
        
        print(f"点数: {actor._xyz.shape[0]}")
        print(f"_features_dc shape: {actor._features_dc.shape}")
        print(f"_features_rest shape: {actor._features_rest.shape}")
        
        # 检查特征统计
        features_dc = actor._features_dc.detach()
        features_rest = actor._features_rest.detach()
        
        print(f"\n_features_dc 统计:")
        print(f"  min: {features_dc.min().item():.6f}")
        print(f"  max: {features_dc.max().item():.6f}")
        print(f"  mean: {features_dc.mean().item():.6f}")
        print(f"  std: {features_dc.std().item():.6f}")
        
        print(f"\n_features_rest 统计:")
        print(f"  min: {features_rest.min().item():.6f}")
        print(f"  max: {features_rest.max().item():.6f}")
        print(f"  mean: {features_rest.mean().item():.6f}")
        print(f"  std: {features_rest.std().item():.6f}")
        
        # 检查是否全零
        if features_rest.abs().sum() < 1e-6:
            print(f"\n⚠️  警告: _features_rest 几乎全零！颜色信息丢失！")
        else:
            print(f"\n✓ _features_rest 有有效数据")
        
        # 检查不透明度
        opacity = actor._opacity.detach()
        print(f"\n不透明度统计:")
        print(f"  min: {opacity.min().item():.6f}")
        print(f"  max: {opacity.max().item():.6f}")
        print(f"  mean: {opacity.mean().item():.6f}")
        
        # 比较与原始对象
        orig_name = obj_name.replace('_sample', '')
        if orig_name in gaussians.obj_list:
            orig_actor = getattr(gaussians, orig_name)
            print(f"\n与 {orig_name} 对比:")
            print(f"  原始对象点数: {orig_actor._xyz.shape[0]}")
            print(f"  原始对象 _features_dc shape: {orig_actor._features_dc.shape}")
            print(f"  原始对象 _features_rest shape: {orig_actor._features_rest.shape}")
            
            if orig_actor._features_dc.shape[1] != actor._features_dc.shape[1]:
                print(f"  ⚠️  fourier_dim 不匹配: {orig_actor._features_dc.shape[1]} vs {actor._features_dc.shape[1]}")
            
            if orig_actor._features_rest.shape[1] != actor._features_rest.shape[1]:
                print(f"  ⚠️  SH 维度不匹配: {orig_actor._features_rest.shape[1]} vs {actor._features_rest.shape[1]}")
    
    print("\n" + "=" * 80)
    print("检查完成")
    print("=" * 80)

if __name__ == "__main__":
    check_sample_colors()

