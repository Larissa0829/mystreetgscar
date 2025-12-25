#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量处理input图片，使用TRELLIS生成ply文件
读取 data/waymo/002/sample/obj_XXX_input.png
生成 obj_XXX_sample.ply
"""

import os
import sys
import glob
import argparse
from PIL import Image

# 添加 TRELLIS目录到python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'TRELLIS'))

# 设置环境变量
os.environ['SPCONV_ALGO'] = 'native'

from trellis.pipelines import TrellisImageTo3DPipeline


def process_input_image(input_path, output_dir, pipeline):
    """
    处理input图片，生成ply文件
    
    Args:
        input_path: input图片路径
        output_dir: 输出目录
        pipeline: TRELLIS pipeline对象
    """
    # 获取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # 从文件名中提取编号（例如从 obj_000_input 中提取 000）
    # 假设格式为 obj_XXX_input，提取 XXX 部分
    if '_' in base_name:
        parts = base_name.split('_')
        # 查找包含数字的部分作为obj_id
        obj_id = None
        for part in parts:
            if part.isdigit():
                obj_id = part
                break
        if obj_id is None:
            # 如果找不到数字，使用最后一个部分（去掉input）
            obj_id = parts[-2] if len(parts) > 1 and parts[-1] == 'input' else parts[-1]
    else:
        obj_id = base_name
    
    # 读取input图片
    input_image = Image.open(input_path)
    
    # 使用TRELLIS生成3D模型
    print(f"正在处理: {base_name}")
    try:
        outputs = pipeline.run(
            input_image,
            seed=1,
            formats=['gaussian'],
        )
        
        # 保存ply文件，格式为 obj_XXX_sample.ply
        ply_path = os.path.join(output_dir, f"obj_{obj_id}_sample.ply")
        outputs['gaussian'][0].save_ply(ply_path)
        print(f"已生成ply文件: {ply_path}")
        
    except Exception as e:
        print(f"处理 {base_name} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量处理input图片，使用TRELLIS生成ply文件')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录路径，例如: data/waymo/002/sample')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 目录 {data_dir} 不存在")
        return
    
    # 加载TRELLIS pipeline
    print("正在加载TRELLIS模型...")
    try:
        pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        pipeline.cuda()
        print("模型加载完成")
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 查找所有input图片文件
    pattern = os.path.join(data_dir, "*_input.png")
    input_files = sorted(glob.glob(pattern))
    
    print(f"找到 {len(input_files)} 个input图片文件")
    
    # 处理每个input图片
    for input_path in input_files:
        process_input_image(input_path, data_dir, pipeline)
    
    print("处理完成！")


if __name__ == "__main__":
    main()

