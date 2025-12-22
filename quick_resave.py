"""
快速重新保存脚本（简化版）
直接修改这里的路径和迭代数，然后运行：python quick_resave.py
"""

from lib.models.scene import Scene
from lib.config import cfg

# ========== 修改这里 ==========
MODEL_PATH = "output/waymo_full_exp/waymo_train_002_exp_origin0"
ITERATION = 30000
# ==============================

print(f"加载模型: {MODEL_PATH}")
print(f"迭代数: {ITERATION}")

cfg.model_path = MODEL_PATH
scene = Scene(cfg.data, cfg.model, load_iteration=ITERATION)

print("\n重新保存...")
scene.save(ITERATION)

print("\n✓ 完成！")
print(f"保存位置: {MODEL_PATH}/point_cloud/iteration_{ITERATION}/")

