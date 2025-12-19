"""
车辆对称性优化工具
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree


def detect_symmetry_axis(xyz):
    """
    自动检测对称轴
    
    Args:
        xyz: [N, 3] 点云坐标
    
    Returns:
        axis: 对称轴索引 (0=X, 1=Y, 2=Z)
        center: 对称中心
    """
    from sklearn.decomposition import PCA
    
    xyz_np = xyz.detach().cpu().numpy()
    
    # 使用 PCA 找主方向
    pca = PCA(n_components=3)
    pca.fit(xyz_np)
    
    # 主方向应该是车头方向（最大方差）
    forward = pca.components_[0]
    
    # 对称轴应该垂直于车头方向
    # 通常车辆沿 Y 轴对称（左右对称）
    # 但我们自动检测
    
    # 计算每个轴的对称性得分
    center = xyz_np.mean(axis=0)
    scores = []
    
    for axis in range(3):
        # 沿该轴镜像
        xyz_mirrored = xyz_np.copy()
        xyz_mirrored[:, axis] = 2 * center[axis] - xyz_mirrored[:, axis]
        
        # 计算原始点云和镜像点云的距离
        tree = cKDTree(xyz_np)
        distances, _ = tree.query(xyz_mirrored, k=1)
        
        # 距离越小，对称性越好
        score = distances.mean()
        scores.append(score)
    
    # 选择对称性最好的轴
    best_axis = np.argmin(scores)
    
    return best_axis, center


def mirror_gaussians(actor, axis=1, confidence_threshold=0.1, use_confidence=True):
    """
    镜像 Gaussian 点云以填补未观测区域（自适应方向）
    
    Args:
        actor: GaussianModelActor 对象
        axis: 对称轴 (0=X, 1=Y, 2=Z)，默认 1 (Y轴，左右对称)
        confidence_threshold: 置信度阈值，低于此值的区域会被镜像点填补
        use_confidence: 是否使用置信度（True=用于原始对象，False=用于sample）
    
    Returns:
        tensors_dict: 新增点的参数字典
    """
    xyz = actor._xyz.data
    
    # 找到左侧和右侧的点
    center = xyz[:, axis].mean()
    left_mask = xyz[:, axis] > center + 0.1   # 左侧
    right_mask = xyz[:, axis] < center - 0.1  # 右侧
    
    if use_confidence:
        # 原始对象：基于置信度判断哪一侧观测更充分
        grad_accum = actor.xyz_gradient_accum
        
        # 确保是一维张量
        if grad_accum.dim() > 1:
            grad_accum = grad_accum.squeeze()
        if grad_accum.dim() > 1:
            grad_accum = grad_accum.mean(dim=-1) if grad_accum.shape[-1] > 1 else grad_accum.squeeze(-1)
        
        confidence = grad_accum / (grad_accum.max() + 1e-8)
        
        # 找到高置信度（观测充分）的点
        left_confident = left_mask & (confidence > confidence_threshold)
        right_confident = right_mask & (confidence > confidence_threshold)
        
        # 自适应决定镜像方向（从观测多的一侧到观测少的一侧）
        if left_confident.sum() > right_confident.sum():
            # 左侧观测多，镜像到右侧
            source_mask = left_confident
            target_side = "右侧"
            print(f"    检测: 左侧观测充分 ({left_confident.sum()} 点)，镜像到右侧")
        else:
            # 右侧观测多，镜像到左侧
            source_mask = right_confident
            target_side = "左侧"
            print(f"    检测: 右侧观测充分 ({right_confident.sum()} 点)，镜像到左侧")
    else:
        # sample 对象：简单的点数判断（不依赖置信度）
        if left_mask.sum() > right_mask.sum():
            # 左侧点多，镜像到右侧
            source_mask = left_mask
            target_side = "右侧"
            print(f"    检测: 左侧点更多 ({left_mask.sum()} 点)，镜像到右侧")
        else:
            # 右侧点多，镜像到左侧
            source_mask = right_mask
            target_side = "左侧"
            print(f"    检测: 右侧点更多 ({right_mask.sum()} 点)，镜像到左侧")
    
    if source_mask.sum() == 0:
        print("    警告: 没有足够的点进行镜像")
        return None
    
    # 镜像点云
    mirrored_xyz = xyz[source_mask].clone()
    mirrored_xyz[:, axis] = 2 * center - mirrored_xyz[:, axis]
    
    # 检查镜像点是否已被覆盖
    # 使用 KDTree 找到最近邻
    tree = cKDTree(xyz.detach().cpu().numpy())
    distances, _ = tree.query(mirrored_xyz.detach().cpu().numpy(), k=1)
    
    # 只保留距离较远的点（未被覆盖的区域）
    uncovered_mask = distances > 0.5  # 阈值可调
    
    if uncovered_mask.sum() == 0:
        print("    信息: 所有镜像点都已被覆盖，无需添加")
        return None
    
    # 获取需要添加的点
    new_xyz = mirrored_xyz[uncovered_mask]
    
    # 复制其他属性
    new_rotation = actor._rotation.data[source_mask][uncovered_mask].clone()
    new_scaling = actor._scaling.data[source_mask][uncovered_mask].clone()
    new_opacity = actor._opacity.data[source_mask][uncovered_mask].clone() * 0.8  # 降低置信度
    new_features_dc = actor._features_dc.data[source_mask][uncovered_mask].clone()
    new_features_rest = actor._features_rest.data[source_mask][uncovered_mask].clone()
    new_semantic = actor._semantic.data[source_mask][uncovered_mask].clone()
    
    # 调整旋转（镜像对称）
    # 四元数镜像：沿 Y 轴镜像 -> [qw, -qx, qy, -qz]
    if axis == 1:  # Y 轴
        new_rotation[:, 1] = -new_rotation[:, 1]  # qx
        new_rotation[:, 3] = -new_rotation[:, 3]  # qz
    elif axis == 0:  # X 轴
        new_rotation[:, 2] = -new_rotation[:, 2]  # qy
        new_rotation[:, 3] = -new_rotation[:, 3]  # qz
    elif axis == 2:  # Z 轴
        new_rotation[:, 1] = -new_rotation[:, 1]  # qx
        new_rotation[:, 2] = -new_rotation[:, 2]  # qy
    
    tensors_dict = {
        "xyz": new_xyz,
        "rotation": new_rotation,
        "scaling": new_scaling,
        "opacity": new_opacity,
        "features_dc": new_features_dc,
        "features_rest": new_features_rest,
        "semantic": new_semantic
    }
    
    return tensors_dict


def compute_symmetry_loss(actor, axis=1, weight=0.01, max_samples=2000):
    """
    计算对称性损失（采样版本，高效）
    
    Args:
        actor: GaussianModelActor 对象
        axis: 对称轴 (0=X, 1=Y, 2=Z)
        weight: 损失权重
        max_samples: 最大采样点数（降低计算量）
    
    Returns:
        loss: 对称性损失
    """
    xyz = actor.get_xyz
    features = actor.get_features
    
    # 找到左侧和右侧的点
    center = xyz[:, axis].mean()
    left_mask = xyz[:, axis] > center + 0.1
    right_mask = xyz[:, axis] < center - 0.1
    
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return torch.tensor(0.0, device=xyz.device)
    
    left_xyz = xyz[left_mask]
    right_xyz = xyz[right_mask]
    
    left_features = features[left_mask]
    right_features = features[right_mask]
    
    # ========== 关键优化：采样 ==========
    # 如果点太多，随机采样以加速
    if left_xyz.shape[0] > max_samples:
        sample_idx = torch.randperm(left_xyz.shape[0], device=xyz.device)[:max_samples]
        left_xyz = left_xyz[sample_idx]
        left_features = left_features[sample_idx]
    
    if right_xyz.shape[0] > max_samples:
        sample_idx = torch.randperm(right_xyz.shape[0], device=xyz.device)[:max_samples]
        right_xyz = right_xyz[sample_idx]
        right_features = right_features[sample_idx]
    # ====================================
    
    # 镜像右侧点到左侧
    right_xyz_mirrored = right_xyz.clone()
    right_xyz_mirrored[:, axis] = 2 * center - right_xyz_mirrored[:, axis]
    
    # ========== 关键优化：分批 GPU 计算 ==========
    # 采样后点数少，可以安全使用 GPU
    batch_size = 500  # 每批处理 500 个点
    num_left = left_xyz.shape[0]
    
    all_min_dist = []
    all_indices = []
    
    for i in range(0, num_left, batch_size):
        end_i = min(i + batch_size, num_left)
        batch_left = left_xyz[i:end_i]
        
        # 计算批次内的距离
        batch_dist = torch.cdist(batch_left, right_xyz_mirrored)  # [batch, N_R]
        batch_min_dist, batch_indices = batch_dist.min(dim=1)
        
        all_min_dist.append(batch_min_dist)
        all_indices.append(batch_indices)
    
    min_dist = torch.cat(all_min_dist)
    indices = torch.cat(all_indices)
    # ===========================================
    
    # 只计算距离较近的点对（真正对称的点）
    close_mask = min_dist < 1.0  # 阈值可调
    
    if close_mask.sum() == 0:
        return torch.tensor(0.0, device=xyz.device)
    
    # 对称点的特征应该相似
    left_feat_close = left_features[close_mask]
    right_feat_close = right_features[indices[close_mask]]
    
    feature_loss = F.mse_loss(left_feat_close, right_feat_close)
    
    # 位置损失（对称点应该对称分布）
    position_loss = min_dist[close_mask].mean()
    
    total_loss = (feature_loss * 0.1 + position_loss * 1.0) * weight
    
    return total_loss


def apply_symmetry_to_all_objects(gaussians, axis=1, mirror=True, add_loss=False):
    """
    对所有动态对象应用对称性优化（原始对象）
    
    Args:
        gaussians: StreetGaussianModel
        axis: 对称轴
        mirror: 是否进行镜像补全
        add_loss: 是否返回对称性损失（用于训练）
    
    Returns:
        loss: 对称性损失（如果 add_loss=True）
    """
    total_loss = 0.0
    mirrored_count = 0
    
    for obj_name in gaussians.obj_list:
        # 跳过背景、天空和 sample 对象
        if obj_name in ['sky', 'background'] or obj_name.endswith('_sample'):
            continue
        
        actor = getattr(gaussians, obj_name)
        
        # 镜像补全
        if mirror:
            tensors_dict = mirror_gaussians(actor, axis=axis)
            if tensors_dict is not None:
                num_new = tensors_dict['xyz'].shape[0]
                actor.densification_postfix(tensors_dict)
                mirrored_count += num_new
                print(f"    ✓ {obj_name}: 添加 {num_new} 个镜像点")
        
        # 计算对称性损失
        if add_loss:
            loss = compute_symmetry_loss(actor, axis=axis)
            total_loss += loss
    
    if mirror and mirrored_count > 0:
        print(f"  ✓ 共添加 {mirrored_count} 个镜像点")
    
    if add_loss:
        return total_loss
    
    return None


def apply_symmetry_to_sample_objects(gaussians, axis=1, mirror=True, add_loss=False):
    """
    对 sample 对象应用对称性优化
    
    Args:
        gaussians: StreetGaussianModel
        axis: 对称轴
        mirror: 是否进行镜像补全
        add_loss: 是否返回对称性损失（用于训练）
    
    Returns:
        loss: 对称性损失（如果 add_loss=True）
    """
    total_loss = 0.0
    mirrored_count = 0
    
    for obj_name in gaussians.obj_list:
        # 只处理 sample 对象
        if not obj_name.endswith('_sample'):
            continue
        
        actor = getattr(gaussians, obj_name)
        
        # 镜像补全
        if mirror:
            tensors_dict = mirror_gaussians(actor, axis=axis)
            if tensors_dict is not None:
                num_new = tensors_dict['xyz'].shape[0]
                actor.densification_postfix(tensors_dict)
                mirrored_count += num_new
                print(f"    ✓ {obj_name}: 添加 {num_new} 个镜像点")
        
        # 计算对称性损失
        if add_loss:
            loss = compute_symmetry_loss(actor, axis=axis)
            total_loss += loss
    
    if mirror and mirrored_count > 0:
        print(f"  ✓ 共添加 {mirrored_count} 个镜像点到 sample 对象")
    
    if add_loss:
        return total_loss
    
    return None

