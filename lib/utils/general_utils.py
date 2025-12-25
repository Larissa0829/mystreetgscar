#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


import numpy as np
import random
import roma
import os
import math
import torch
import torch.nn.functional as F
import sys
import cv2
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from lib.utils.graphics_utils import focal2fov
from lib.datasets.base_readers import CameraInfo

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution=None, resize_mode=Image.BILINEAR):
    if resolution is not None:
        resized_image_PIL = pil_image.resize(resolution, resize_mode)
    
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    
def NumpytoTorch(image, resolution, resize_mode=cv2.INTER_AREA):
    if resolution is not None:
        image = cv2.resize(image, resolution, interpolation=resize_mode)
    
    image = torch.from_numpy(np.array(image))
    if len(image.shape) == 2:
        image = image[..., None].permute(2, 0, 1) # [1, H, W]
    elif len(image.shape) == 3:
        image = image.permute(2, 0, 1)
    
    return image

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000, warmup_steps=0
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0) or (step < warmup_steps):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def quaternion_to_matrix_numpy(r):
    q = r / np.linalg.norm(r)

    R = np.zeros((3, 3))

    r = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    R[0, 0] = 1 - 2 * (y*y + z*z)
    R[0, 1] = 2 * (x*y - r*z)
    R[0, 2] = 2 * (x*z + r*y)
    R[1, 0] = 2 * (x*y + r*z)
    R[1, 1] = 1 - 2 * (x*x + z*z)
    R[1, 2] = 2 * (y*z - r*x)
    R[2, 0] = 2 * (x*z - r*y)
    R[2, 1] = 2 * (y*z + r*x)
    R[2, 2] = 1 - 2 * (x*x + y*y)
    return R


def quaternion_to_matrix(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    
def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling

def quaternion_slerp(q0: torch.Tensor, q1: torch.Tensor, step=0.5) -> torch.Tensor:
    # https://github.com/clemense/quaternion-conventions
    # 3D Gaussian Format: w-x-y-z Roma Format: x-y-z-w

    ndim = q0.ndim
    if ndim == 1:
        q0 = q0.unsqueeze(0)
        q1 = q1.unsqueeze(0)
        
    q0 = torch.nn.functional.normalize(q0)
    q1 = torch.nn.functional.normalize(q1)
    q0 = q0[..., [1, 2, 3, 0]]
    q1 = q1[..., [1, 2, 3, 0]]
    steps = torch.tensor([step], device=q1.device).float()
    q = roma.utils.unitquat_slerp(q0, q1, steps) 
    q = q[..., [3, 0, 1, 2]].squeeze(0)
    
    if ndim == 1:
        q = q.squeeze(0)
    
    return q

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = quaternion_to_matrix(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    # sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # torch.cuda.set_device(torch.device("cuda:0"))

def startswith_any(k, l):
    for s in l:
        if k.startswith(s):
            return True
    return False

# We make an exception on snake case conventions because SO3 != so3.
def exp_map_SO3xR3(tangent_vector):
    """Compute the exponential map of the direct product group `SO(3) x R^3`.

    This can be used for learning pose deltas on SE(3), and is generally faster than `exp_map_SE3`.

    Args:
        tangent_vector: Tangent vector; length-3 translations, followed by an `so(3)` tangent vector.
    Returns:
        [R|t] transformation matrices.
    """
    # code for SO3 map grabbed from pytorch3d and stripped down to bare-bones
    log_rot = tangent_vector[:, 3:]
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, 1e-4).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = torch.zeros((log_rot.shape[0], 3, 3), dtype=log_rot.dtype, device=log_rot.device)
    skews[:, 0, 1] = -log_rot[:, 2]
    skews[:, 0, 2] = log_rot[:, 1]
    skews[:, 1, 0] = log_rot[:, 2]
    skews[:, 1, 2] = -log_rot[:, 0]
    skews[:, 2, 0] = -log_rot[:, 1]
    skews[:, 2, 1] = log_rot[:, 0]
    skews_square = torch.bmm(skews, skews)

    ret = torch.zeros(tangent_vector.shape[0], 3, 4, dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    # Compute the translation
    ret[:, :3, 3] = tangent_vector[:, :3]
    return ret

def matrix_to_axis_angle(matrix):
    R = matrix[..., :3, :3]
    T = matrix[..., :3, 3]
    R = quaternion_to_axis_angle(matrix_to_quaternion(R))
    matrix = torch.cat([R, T], dim=-1)
    
    return matrix

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def euler_angles_to_matrix(euler_angles, convention="XYZ"):
    """
    Convert Euler angles to a rotation matrix.
    
    Args:
        euler_angles (torch.Tensor): Tensor of shape (N, 3), where each row contains
                                     the Euler angles [angle_x, angle_y, angle_z] in radians.
        convention (str): Rotation order, e.g., "XYZ", "ZYX".
    
    Returns:
        torch.Tensor: Rotation matrices of shape (N, 3, 3).
    """
    if euler_angles.ndim != 2 or euler_angles.shape[1] != 3:
        raise ValueError("euler_angles must be of shape (N, 3)")

    # Extract angles
    x = euler_angles[:, 0]
    y = euler_angles[:, 1]
    z = euler_angles[:, 2]

    # Compute individual rotation matrices
    cos_x, sin_x = torch.cos(x), torch.sin(x)
    cos_y, sin_y = torch.cos(y), torch.sin(y)
    cos_z, sin_z = torch.cos(z), torch.sin(z)

    R_x = torch.stack([
        torch.stack([torch.ones_like(cos_x), torch.zeros_like(cos_x), torch.zeros_like(cos_x)], dim=-1),
        torch.stack([torch.zeros_like(cos_x), cos_x, -sin_x], dim=-1),
        torch.stack([torch.zeros_like(cos_x), sin_x, cos_x], dim=-1)
    ], dim=1)

    R_y = torch.stack([
        torch.stack([cos_y, torch.zeros_like(cos_y), sin_y], dim=-1),
        torch.stack([torch.zeros_like(cos_y), torch.ones_like(cos_y), torch.zeros_like(cos_y)], dim=-1),
        torch.stack([-sin_y, torch.zeros_like(cos_y), cos_y], dim=-1)
    ], dim=1)

    R_z = torch.stack([
        torch.stack([cos_z, -sin_z, torch.zeros_like(cos_z)], dim=-1),
        torch.stack([sin_z, cos_z, torch.zeros_like(cos_z)], dim=-1),
        torch.stack([torch.zeros_like(cos_z), torch.zeros_like(cos_z), torch.ones_like(cos_z)], dim=-1)
    ], dim=1)

    # Combine rotations based on the convention
    if convention == "XYZ":
        rotation_matrix = torch.matmul(torch.matmul(R_z, R_y), R_x)
    elif convention == "ZYX":
        rotation_matrix = torch.matmul(torch.matmul(R_x, R_y), R_z)
    else:
        raise ValueError(f"Unsupported rotation convention: {convention}")

    return rotation_matrix

    
def draw_bbox_for_objects(gaussians,viewpoint_cam,render_img_tensor,origin_list,color_list=None,):
    """
    render_img_tensor: [3, H, W], RGB, torch tensor
    """
    img = (
        render_img_tensor.clamp(0, 1)
        .detach().cpu().numpy() * 255
    ).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    H, W = img.shape[:2]
    drawn = 0

    for obj_name in origin_list:

        if obj_name in ["sky", "background"]:
            continue
        if not hasattr(gaussians, obj_name):
            continue

        obj = getattr(gaussians, obj_name)

        # ---------- timestamp ----------
        t = viewpoint_cam.meta["timestamp"]
        if t < obj.start_timestamp or t > obj.end_timestamp:
            continue

        # ======================================================
        # 1. 使用obj_meta中的bounding box信息（而不是点云的）
        # ======================================================
        length, width, height = obj.obj_meta['length'], obj.obj_meta['width'], obj.obj_meta['height']
        
        # 在obj的局部坐标系中定义8个角点
        # 坐标系：车头朝x正轴，向上为z正轴
        # bbox中心在原点，尺寸为 [length, width, height] = [X, Y, Z]
        half_length = length / 2.0
        half_width = width / 2.0
        half_height = height / 2.0
        
        corners_local = torch.tensor([
            [-half_length, -half_width, -half_height],  # 0: 左下后
            [ half_length, -half_width, -half_height],  # 1: 右下后
            [-half_length,  half_width, -half_height],  # 2: 左上后
            [ half_length,  half_width, -half_height],  # 3: 右上后
            [-half_length, -half_width,  half_height],  # 4: 左下前
            [ half_length, -half_width,  half_height],  # 5: 右下前
            [-half_length,  half_width,  half_height],  # 6: 左上前
            [ half_length,  half_width,  half_height],  # 7: 右上前
        ], device='cuda', dtype=torch.float32)
        
        # ======================================================
        # 2. 获取obj在世界坐标系中的pose（通过actor_pose）
        # ======================================================
        # 需要先设置visibility并parse_camera，才能获取obj_rots和obj_trans
        if not hasattr(gaussians, 'graph_obj_list') or obj_name not in gaussians.graph_obj_list:
            # 如果obj不在graph_obj_list中，需要重新设置visibility并parse_camera
            gaussians.set_visibility(origin_list)
            gaussians.parse_camera(viewpoint_cam, custom_rotation=None, custom_translation=None)
        
        if obj_name not in gaussians.graph_obj_list:
            continue
        
        # 获取obj在graph_gaussian_range中的起始索引
        if obj_name not in gaussians.graph_gaussian_range:
            continue
        
        start_idx = gaussians.graph_gaussian_range[obj_name][0]
        
        # 获取parse_camera中已经计算好的旋转和平移（已考虑ego pose）
        obj_rot = gaussians.obj_rots[start_idx]  # [4] 四元数
        obj_trans = gaussians.obj_trans[start_idx]  # [3] 平移向量
        
        # ======================================================
        # 3. 将局部坐标转换为世界坐标
        # ======================================================
        # 将旋转四元数转换为旋转矩阵
        if obj_rot.dim() == 1:
            obj_rot = obj_rot.unsqueeze(0)  # [1, 4]
        rot_matrix = quaternion_to_matrix(obj_rot).squeeze(0)  # [3, 3]
        
        # 将局部坐标转换为世界坐标
        corners_world = (rot_matrix @ corners_local.T).T + obj_trans.unsqueeze(0)  # [8, 3]

        # ======================================================
        # 6. World → Camera
        # ======================================================
        ones = torch.ones((8, 1), device=xyz.device)
        corners_h = torch.cat([corners_world, ones], dim=1)

        cam = (viewpoint_cam.world_view_transform @ corners_h.T).T
        if not (cam[:, 2] < -1e-3).any():
            continue

        # ======================================================
        # 7. Projection
        # ======================================================
        proj = (viewpoint_cam.full_proj_transform @ corners_h.T).T
        xy = proj[:, :2] / proj[:, 3:4]

        u = (xy[:, 0] + 1) * 0.5 * W
        v = (1 - xy[:, 1]) * 0.5 * H

        pts = torch.stack([u, v], dim=1).cpu().numpy().astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

        # ======================================================
        # 8. Draw
        # ======================================================
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7),
        ]

        for i, j in edges:
            cv2.line(img, tuple(pts[i]), tuple(pts[j]), (0,255,0), 2)

        drawn += 1

    print(f"[BBox] drawn {drawn}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2,0,1).float().cuda() / 255.0
    return img