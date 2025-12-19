#!/usr/bin/env python3
"""
为已训练的模型添加对称性优化
适用于已经训练完成但忘记启用对称性的情况
"""

import torch
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

from lib.models.scene import Scene
from lib.config import cfg

def add_symmetry_to_model(model_path, iteration):
    """
    为已训练的模型添加对称性
    
    Args:
        model_path: 模型路径
        iteration: 迭代数
    """
    print("=" * 80)
    print("为已训练模型添加对称性优化")
    print("=" * 80)
    print(f"  模型路径: {model_path}")
    print(f"  迭代数: {iteration}")
    print("=" * 80)
    
    # 设置配置
    cfg.model_path = model_path
    
    # 加载场景
    print("\n[1/3] 加载场景...")
    scene = Scene(cfg.data, cfg.model, load_iteration=iteration)
    gaussians = scene.gaussians
    
    print(f"  ✓ 加载完成")
    print(f"  对象数: {gaussians.models_num}")
    print(f"  对象列表: {gaussians.obj_list}")
    
    # 检查是否有 sample 对象
    sample_objects = [name for name in gaussians.obj_list if name.endswith('_sample')]
    if not sample_objects:
        print("\n  ⚠️ 警告: 没有找到 sample 对象")
        print("  请确保已经运行过 TRELLIS 生成（iteration 16001）")
        return
    
    print(f"\n  找到 {len(sample_objects)} 个 sample 对象:")
    for name in sample_objects:
        actor = getattr(gaussians, name)
        print(f"    - {name}: {actor._xyz.shape[0]} 点")
    
    # 应用对称性镜像补全
    print("\n[2/3] 应用对称性优化...")
    from lib.utils.symmetry_utils import apply_symmetry_to_sample_objects
    
    apply_symmetry_to_sample_objects(
        gaussians, 
        axis=1,  # Y轴对称（左右对称）
        mirror=True,  # 进行镜像补全
        add_loss=False  # 不计算损失
    )
    
    print("\n  对称性优化后的点数:")
    for name in sample_objects:
        actor = getattr(gaussians, name)
        print(f"    - {name}: {actor._xyz.shape[0]} 点")
    
    # 保存改进后的模型
    print("\n[3/3] 保存改进后的模型...")
    scene.save(iteration)
    
    print("\n" + "=" * 80)
    print("✓ 对称性优化完成！")
    print(f"  保存位置: {model_path}/point_cloud/iteration_{iteration}/")
    print("=" * 80)
    print("\n提示:")
    print("  1. 检查点云文件，应该看到点数增加")
    print("  2. 可以继续训练，或直接使用改进后的模型")
    print("  3. 如果要继续训练，建议从当前 iteration 恢复")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="为已训练模型添加对称性优化")
    parser.add_argument("--model_path", type=str, 
                       default="output/waymo_full_exp/waymo_train_002_exp_origin0",
                       help="模型路径")
    parser.add_argument("--iteration", type=int, default=30000,
                       help="要加载的迭代数")
    
    args = parser.parse_args()
    
    # 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        sys.exit(1)
    
    # 检查 checkpoint 是否存在
    checkpoint_path = os.path.join(args.model_path, f"trained_model/iteration_{args.iteration}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"错误: Checkpoint 不存在: {checkpoint_path}")
        print(f"可用的 checkpoints:")
        trained_model_dir = os.path.join(args.model_path, "trained_model")
        if os.path.exists(trained_model_dir):
            checkpoints = [f for f in os.listdir(trained_model_dir) if f.endswith('.pth')]
            for ckpt in sorted(checkpoints):
                print(f"  - {ckpt}")
        sys.exit(1)
    
    add_symmetry_to_model(args.model_path, args.iteration)

