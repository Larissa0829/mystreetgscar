import torch
import torch.nn as nn
import numpy as np
import os
from simple_knn._C import distCUDA2
from lib.config import cfg
from lib.utils.general_utils import quaternion_to_matrix, \
    build_scaling_rotation, \
    strip_symmetric, \
    quaternion_raw_multiply, \
    startswith_any, \
    matrix_to_quaternion, \
    euler_angles_to_matrix, \
    quaternion_invert
from lib.utils.graphics_utils import BasicPointCloud
from lib.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from lib.models.gaussian_model import GaussianModel
from lib.models.gaussian_model_bkgd import GaussianModelBkgd
from lib.models.gaussian_model_actor import GaussianModelActor
from lib.models.gaussian_model_sky import GaussinaModelSky
from bidict import bidict
from lib.utils.camera_utils import Camera
from lib.utils.sh_utils import eval_sh
from lib.models.actor_pose import ActorPose
from lib.models.sky_cubemap import SkyCubeMap
from lib.models.color_correction import ColorCorrection
from lib.models.camera_pose import PoseCorrection


class StreetGaussianModel(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata
            
        self.max_sh_degree = cfg.model.gaussian.sh_degree
        self.active_sh_degree = self.max_sh_degree

        # background + moving objects
        self.include_background = cfg.model.nsg.get('include_bkgd', True)
        self.include_obj = cfg.model.nsg.get('include_obj', True)
        
        # sky (modeling sky with gaussians, if set to false represent the sky with cube map)
        self.include_sky = cfg.model.nsg.get('include_sky', False) 
        if self.include_sky:
            assert cfg.data.white_background is False

                
        # fourier sh dimensions
        self.fourier_dim = cfg.model.gaussian.get('fourier_dim', 1)
        
        # layer color correction
        self.use_color_correction = cfg.model.use_color_correction
        
        # camera pose optimizations (not test)
        self.use_pose_correction = cfg.model.use_pose_correction
    
        # symmetry
        self.flip_prob = cfg.model.gaussian.get('flip_prob', 0.)
        self.flip_axis = 1 
        self.flip_matrix = torch.eye(3).float().cuda() * -1
        self.flip_matrix[self.flip_axis, self.flip_axis] = 1
        self.flip_matrix = matrix_to_quaternion(self.flip_matrix.unsqueeze(0))
        self.setup_functions() 
    
    def set_visibility(self, include_list):
        self.include_list = include_list # prefix

    def get_visibility(self, model_name):
        if model_name == 'background':
            if model_name in self.include_list and self.include_background:
                return True
            else:
                return False
        elif model_name == 'sky':
            if model_name in self.include_list and self.include_sky:
                return True
            else:
                return False
        elif model_name.startswith('obj_'):
            if model_name in self.include_list and self.include_obj:
                return True
            else:
                return False
        else:
            raise ValueError(f'Unknown model name {model_name}')
                
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            if model_name in ['background', 'sky']:
                model.create_from_pcd(pcd, spatial_lr_scale)
            else:
                model.create_from_pcd(spatial_lr_scale)

    def save_ply(self, path):

        base_dir = os.path.dirname(path)
        mkdir_p(base_dir)
        
        plydata_list = []
        for i in range(self.models_num):
            model_name = self.model_name_id.inverse[i]
            model: GaussianModel = getattr(self, model_name)
            data = model.make_ply()
            plydata = PlyElement.describe(data, f'vertex_{model_name}')
            plydata_list.append(plydata)

            # 动态对象单独保存
            if model_name != 'background' and not model_name.endswith('_sample'):
                # 检查是否有对应的 sample 对象
                sample_name = f"{model_name}_sample"
                if hasattr(self, sample_name) and sample_name in self.model_name_id:
                    # 存在 sample 对象，合并保存
                    sample_model: GaussianModel = getattr(self, sample_name)
                    sample_data = sample_model.make_ply()
                    
                    # 检查 dtype 是否一致并合并
                    import numpy as np
                    if data.dtype == sample_data.dtype:
                        # dtype 一致，直接拼接
                        merged_data = np.concatenate([data, sample_data])
                    else:
                        # dtype 不一致，需要统一格式
                        print(f"  警告: {sample_name} 的 dtype 与 {model_name} 不一致，正在转换...")
                        
                        # 创建与原始对象相同 dtype 的空数组
                        merged_data = np.empty(len(data) + len(sample_data), dtype=data.dtype)
                        
                        # 复制原始数据
                        merged_data[:len(data)] = data
                        
                        # 复制 sample 数据（字段对字段）
                        for field_name in data.dtype.names:
                            if field_name in sample_data.dtype.names:
                                merged_data[field_name][len(data):] = sample_data[field_name]
                            else:
                                # 如果 sample 中没有该字段，填充默认值
                                print(f"    警告: sample 缺少字段 {field_name}，填充为 0")
                                merged_data[field_name][len(data):] = 0
                    
                    plydata = PlyElement.describe(merged_data, f'vertex')
                    obj_path = os.path.join(base_dir, f"{model_name}.ply")
                    PlyData([plydata]).write(obj_path)
                    print(f"  ✓ 保存合并点云: {model_name} ({len(data)} 点) + {sample_name} ({len(sample_data)} 点) = {len(merged_data)} 点")
                else:
                    # 没有 sample 对象，单独保存原始点云
                    plydata = PlyElement.describe(data, f'vertex')
                    obj_path = os.path.join(base_dir, f"{model_name}.ply")
                    PlyData([plydata]).write(obj_path)

        PlyData(plydata_list).write(path)
        
    def load_ply(self, path):
        plydata_list = PlyData.read(path).elements
        for plydata in plydata_list:
            model_name = plydata.name[7:] # vertex_.....
            if model_name in self.model_name_id.keys():
                print('Loading model', model_name)
                model: GaussianModel = getattr(self, model_name)
                model.load_ply(path=None, input_ply=plydata)
                plydata_list = PlyData.read(path).elements
                
        self.active_sh_degree = self.max_sh_degree

    def load_sample_ply(self, path):
        plydata = PlyData.read(path).elements[0]
        model_name = "sample"
        print('Loading model', model_name)
        model: GaussianModelActor = getattr(self, model_name)
        model.load_ply(path=None, input_ply=plydata)
        self.active_sh_degree = self.max_sh_degree
    
    def _load_sample_objects_from_ply(self):
        """
        自动检测并加载 sample 点云对象
        在从 checkpoint 加载时调用，确保 sample 对象被正确注册
        """
        if not self.include_obj:
            return
        
        ply_dir = os.path.join(cfg.model_path, 'input_ply')
        if not os.path.exists(ply_dir):
            return
        
        print("\n检测 sample 点云文件...")
        loaded_samples = []
        
        # 遍历所有原始对象
        for obj_name in list(self.obj_list):  # 使用 list() 复制，避免在迭代时修改
            if obj_name in ['sky', 'background'] or obj_name.endswith('_sample'):
                continue
            
            sample_name = f"{obj_name}_sample"
            sample_ply_path = os.path.join(ply_dir, f"{sample_name}.ply")
            
            # 检查 sample ply 文件是否存在
            if not os.path.exists(sample_ply_path):
                continue
            
            # 检查是否已经注册
            if hasattr(self, sample_name) and sample_name in self.model_name_id:
                print(f"  - {sample_name}: 已注册，跳过")
                continue
            
            # 获取原始对象的元数据
            actor: GaussianModelActor = getattr(self, obj_name)
            obj_meta = actor.obj_meta
            
            # 创建新的 sample 对象
            sample_actor = GaussianModelActor(model_name=sample_name, obj_meta=obj_meta)
            
            # 加载 PLY 文件
            sample_actor.load_ply(sample_ply_path)
            
            # 设置 track_id（与原始对象相同）
            sample_actor.track_id = actor.track_id
            
            # 调整特征维度以匹配原始对象（必须在 training_setup 之前）
            actor_fourier_dim = actor._features_dc.shape[1]
            actor_sh_rest_dim = actor._features_rest.shape[1]
            
            sample_fourier_dim = sample_actor._features_dc.shape[1]
            sample_sh_rest_dim = sample_actor._features_rest.shape[1] if sample_actor._features_rest is not None else 0
            
            # 调整 _features_dc
            if sample_fourier_dim != actor_fourier_dim:
                sample_dc = sample_actor._features_dc.data.cuda()
                if sample_fourier_dim < actor_fourier_dim:
                    sample_dc_new = sample_dc[:, :1, :].expand(-1, actor_fourier_dim, -1).clone()
                else:
                    sample_dc_new = sample_dc[:, :actor_fourier_dim, :]
                sample_actor._features_dc = torch.nn.Parameter(sample_dc_new.requires_grad_(True))
            else:
                sample_actor._features_dc = torch.nn.Parameter(sample_actor._features_dc.data.cuda().requires_grad_(True))
            
            # 调整 _features_rest
            if sample_actor._features_rest is None or sample_sh_rest_dim != actor_sh_rest_dim:
                num_points = sample_actor._xyz.shape[0]
                sample_rest_new = torch.zeros((num_points, actor_sh_rest_dim, 3), device='cuda')
                sample_actor._features_rest = torch.nn.Parameter(sample_rest_new.requires_grad_(True))
            else:
                sample_actor._features_rest = torch.nn.Parameter(sample_actor._features_rest.data.cuda().requires_grad_(True))
            
            # 确保 sh_degree 一致
            sample_actor.max_sh_degree = actor.max_sh_degree
            sample_actor.active_sh_degree = actor.active_sh_degree
            
            # 确保其他参数在 CUDA 上
            sample_actor._xyz = torch.nn.Parameter(sample_actor._xyz.data.cuda().requires_grad_(True))
            sample_actor._rotation = torch.nn.Parameter(sample_actor._rotation.data.cuda().requires_grad_(True))
            sample_actor._scaling = torch.nn.Parameter(sample_actor._scaling.data.cuda().requires_grad_(True))
            sample_actor._opacity = torch.nn.Parameter(sample_actor._opacity.data.cuda().requires_grad_(True))
            sample_actor._semantic = torch.nn.Parameter(sample_actor._semantic.data.cuda().requires_grad_(True))
            
            # 初始化训练参数（optimizer, 辅助张量等）
            sample_actor.training_setup()
            
            # 确保所有辅助张量在 CUDA 上
            sample_actor.max_radii2D = torch.zeros(sample_actor._xyz.shape[0], dtype=torch.float32, device='cuda')
            sample_actor.xyz_gradient_accum = sample_actor.xyz_gradient_accum.cuda()
            sample_actor.denom = sample_actor.denom.cuda()
            
            # 注册到 gaussians
            setattr(self, sample_name, sample_actor)
            self.model_name_id[sample_name] = self.models_num
            self.obj_list.append(sample_name)
            self.models_num += 1
            
            loaded_samples.append(sample_name)
            print(f"  ✓ {sample_name}: 已加载并注册 ({sample_actor._xyz.shape[0]} 点)")
        
        if loaded_samples:
            print(f"\n✓ 共加载 {len(loaded_samples)} 个 sample 对象")
        else:
            print("  未发现新的 sample 点云文件")
  
    def load_state_dict(self, state_dict, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.load_state_dict(state_dict[model_name])
        
        if self.actor_pose is not None:
            self.actor_pose.load_state_dict(state_dict['actor_pose'])
            
        if self.sky_cubemap is not None:
            self.sky_cubemap.load_state_dict(state_dict['sky_cubemap'])
            
        if self.color_correction is not None:
            self.color_correction.load_state_dict(state_dict['color_correction'])
            
        if self.pose_correction is not None:
            self.pose_correction.load_state_dict(state_dict['pose_correction'])
        
        # 自动检测并加载 sample 点云（如果存在）
        self._load_sample_objects_from_ply()
                            
    def save_state_dict(self, is_final, exclude_list=[]):
        state_dict = dict()

        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            state_dict[model_name] = model.state_dict(is_final)
        
        if self.actor_pose is not None:
            state_dict['actor_pose'] = self.actor_pose.save_state_dict(is_final)
      
        if self.sky_cubemap is not None:
            state_dict['sky_cubemap'] = self.sky_cubemap.save_state_dict(is_final)
      
        if self.color_correction is not None:
            state_dict['color_correction'] = self.color_correction.save_state_dict(is_final)
      
        if self.pose_correction is not None:
            state_dict['pose_correction'] = self.pose_correction.save_state_dict(is_final)
      
        return state_dict
        
    def setup_functions(self):
        obj_tracklets = self.metadata['obj_tracklets']
        obj_info = self.metadata['obj_meta']
        tracklet_timestamps = self.metadata['tracklet_timestamps']
        camera_timestamps = self.metadata['camera_timestamps']
        
        self.model_name_id = bidict()
        self.obj_list = []
        self.models_num = 0
        self.obj_info = obj_info
        
        # Build background model
        if self.include_background:
            self.background = GaussianModelBkgd(
                model_name='background', 
                scene_center=self.metadata['scene_center'],
                scene_radius=self.metadata['scene_radius'],
                sphere_center=self.metadata['sphere_center'],
                sphere_radius=self.metadata['sphere_radius'],
            )
                                    
            self.model_name_id['background'] = 0
            self.models_num += 1
        
        # Build object model
        if self.include_obj:
            for track_id, obj_meta in self.obj_info.items():
                model_name = f'obj_{track_id:03d}'
                setattr(self, model_name, GaussianModelActor(model_name=model_name, obj_meta=obj_meta))
                self.model_name_id[model_name] = self.models_num
                self.obj_list.append(model_name)
                self.models_num += 1
                
        # Build sky model
        if self.include_sky:
            self.sky_cubemap = SkyCubeMap()    
        else:
            self.sky_cubemap = None    
                             
        # Build actor model 
        if self.include_obj:
            self.actor_pose = ActorPose(obj_tracklets, tracklet_timestamps, camera_timestamps, obj_info)
        else:
            self.actor_pose = None

        # Build color correction
        if self.use_color_correction:
            self.color_correction = ColorCorrection(self.metadata)
        else:
            self.color_correction = None
            
        # Build pose correction
        if self.use_pose_correction:
            self.pose_correction = PoseCorrection(self.metadata)
        else:
            self.pose_correction = None
            
        
    def parse_camera(self, camera: Camera, custom_rotation=None, custom_translation=None):
        # set camera
        self.viewpoint_camera = camera
        
        # set background mask
        self.background.set_background_mask(camera)
        
        self.frame = camera.meta['frame']
        self.frame_idx = camera.meta['frame_idx']
        self.frame_is_val = camera.meta['is_val']
        self.num_gaussians = 0

        # background        
        if self.get_visibility('background'):
            num_gaussians_bkgd = self.background.get_xyz.shape[0]
            self.num_gaussians += num_gaussians_bkgd

        # object (build scene graph)
        self.graph_obj_list = []

        if self.include_obj:
            timestamp = camera.meta['timestamp']
            for i, obj_name in enumerate(self.obj_list):
                obj_model: GaussianModelActor = getattr(self, obj_name)
                start_timestamp, end_timestamp = obj_model.start_timestamp, obj_model.end_timestamp
                if timestamp >= start_timestamp and timestamp <= end_timestamp and self.get_visibility(obj_name):
                    self.graph_obj_list.append(obj_name)
                    num_gaussians_obj = getattr(self, obj_name).get_xyz.shape[0]
                    self.num_gaussians += num_gaussians_obj

        # set index range
        self.graph_gaussian_range = dict()
        idx = 0
        
        if self.get_visibility('background'):
            num_gaussians_bkgd = self.background.get_xyz.shape[0]
            self.graph_gaussian_range['background'] = [idx, idx+num_gaussians_bkgd-1]
            idx += num_gaussians_bkgd
        
        for obj_name in self.graph_obj_list:
            num_gaussians_obj = getattr(self, obj_name).get_xyz.shape[0]
            self.graph_gaussian_range[obj_name] = [idx, idx+num_gaussians_obj-1]
            idx += num_gaussians_obj

        if len(self.graph_obj_list) > 0:
            self.obj_rots = []
            self.obj_trans = []
            for i, obj_name in enumerate(self.graph_obj_list):
                obj_model: GaussianModelActor = getattr(self, obj_name)
                track_id = obj_model.track_id
                obj_rot = self.actor_pose.get_tracking_rotation(track_id, self.viewpoint_camera)
                obj_trans = self.actor_pose.get_tracking_translation(track_id, self.viewpoint_camera)                
                ego_pose = self.viewpoint_camera.ego_pose
                ego_pose_rot = matrix_to_quaternion(ego_pose[:3, :3].unsqueeze(0)).squeeze(0)
                obj_rot = quaternion_raw_multiply(ego_pose_rot.unsqueeze(0), obj_rot.unsqueeze(0)).squeeze(0)

                if custom_rotation is not None: #改变车辆的朝向
                    # 自定义绕自车Z轴旋转
                    yaw_angle = torch.tensor([custom_rotation], device=obj_rot.device) # custom_rotation类似于90°：torch.pi / 2 （弧度）
                    zero = torch.zeros_like(yaw_angle)
                    euler = torch.stack([zero, zero, yaw_angle], dim=1)  # [1, 3] (x, y, z Euler angles)
                    R_custom = euler_angles_to_matrix(euler, convention="XYZ")  # 或 "ZYX"，根据约定调整
                    q_custom = matrix_to_quaternion(R_custom).squeeze(0)

                    # 将自定义旋转应用到当前 obj_rot（右乘：局部旋转）
                    obj_rot = quaternion_raw_multiply(obj_rot.unsqueeze(0), q_custom.unsqueeze(0)).squeeze(0)
                if custom_translation is not None: #改变车辆的位置
                    obj_trans += torch.cat([torch.zeros(1, device=custom_translation.device), custom_translation, torch.zeros(1, device=custom_translation.device)]).cuda()

                obj_trans = ego_pose[:3, :3] @ obj_trans + ego_pose[:3, 3]
                
                obj_rot = obj_rot.expand(obj_model.get_xyz.shape[0], -1)
                obj_trans = obj_trans.unsqueeze(0).expand(obj_model.get_xyz.shape[0], -1)
                
                self.obj_rots.append(obj_rot)
                self.obj_trans.append(obj_trans)
            
            self.obj_rots = torch.cat(self.obj_rots, dim=0)
            self.obj_trans = torch.cat(self.obj_trans, dim=0)  
            
            self.flip_mask = []
            for obj_name in self.graph_obj_list:
                obj_model: GaussianModelActor = getattr(self, obj_name)
                if obj_model.deformable or self.flip_prob == 0:
                    flip_mask = torch.zeros_like(obj_model.get_xyz[:, 0]).bool()
                else:
                    flip_mask = torch.rand_like(obj_model.get_xyz[:, 0]) < self.flip_prob
                self.flip_mask.append(flip_mask)
            self.flip_mask = torch.cat(self.flip_mask, dim=0)   
            
    @property
    def get_scaling(self):
        scalings = []
        
        if self.get_visibility('background'):
            scaling_bkgd = self.background.get_scaling
            scalings.append(scaling_bkgd)
        
        for obj_name in self.graph_obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)

            scaling = obj_model.get_scaling
            
            scalings.append(scaling)
        
        scalings = torch.cat(scalings, dim=0)
        return scalings
            
    @property
    def get_rotation(self):
        rotations = []

        if self.get_visibility('background'):            
            rotations_bkgd = self.background.get_rotation
            if self.use_pose_correction:
                rotations_bkgd = self.pose_correction.correct_gaussian_rotation(self.viewpoint_camera, rotations_bkgd)            
            rotations.append(rotations_bkgd)

        if len(self.graph_obj_list) > 0:
            rotations_local = []
            for i, obj_name in enumerate(self.graph_obj_list):
                obj_model: GaussianModelActor = getattr(self, obj_name)
                rotation_local = obj_model.get_rotation
                rotations_local.append(rotation_local)

            rotations_local = torch.cat(rotations_local, dim=0)
            rotations_local = rotations_local.clone()
            rotations_local[self.flip_mask] = quaternion_raw_multiply(self.flip_matrix, rotations_local[self.flip_mask])
            rotations_obj = quaternion_raw_multiply(self.obj_rots, rotations_local)
            rotations_obj = torch.nn.functional.normalize(rotations_obj)
            rotations.append(rotations_obj)

        rotations = torch.cat(rotations, dim=0)
        return rotations
    
    @property
    def get_xyz(self):
        xyzs = []
        if self.get_visibility('background'):
            xyz_bkgd = self.background.get_xyz
            if self.use_pose_correction:
                xyz_bkgd = self.pose_correction.correct_gaussian_xyz(self.viewpoint_camera, xyz_bkgd)
            
            xyzs.append(xyz_bkgd)
        
        if len(self.graph_obj_list) > 0:
            xyzs_local = []

            for i, obj_name in enumerate(self.graph_obj_list):
                obj_model: GaussianModelActor = getattr(self, obj_name)
                xyz_local = obj_model.get_xyz
                xyzs_local.append(xyz_local)
                
            xyzs_local = torch.cat(xyzs_local, dim=0)
            xyzs_local = xyzs_local.clone()
            xyzs_local[self.flip_mask, self.flip_axis] *= -1
            obj_rots = quaternion_to_matrix(self.obj_rots)
            xyzs_obj = torch.einsum('bij, bj -> bi', obj_rots, xyzs_local) + self.obj_trans
            xyzs.append(xyzs_obj)

        xyzs = torch.cat(xyzs, dim=0)

        return xyzs            

    @property
    def get_features(self):                
        features = []

        if self.get_visibility('background'):
            features_bkgd = self.background.get_features
            features.append(features_bkgd)            
        
        for i, obj_name in enumerate(self.graph_obj_list):
            obj_model: GaussianModelActor = getattr(self, obj_name)
            feature_obj = obj_model.get_features_fourier(self.frame)
            features.append(feature_obj)
            
        features = torch.cat(features, dim=0)
       
        return features
    
    def get_colors(self, camera_center):
        colors = []

        model_names = []
        if self.get_visibility('background'):
            model_names.append('background')

        model_names.extend(self.graph_obj_list)

        for model_name in model_names:
            if model_name == 'background':                
                model: GaussianModel= getattr(self, model_name)
            else:
                model: GaussianModelActor = getattr(self, model_name)
                
            max_sh_degree = model.max_sh_degree
            sh_dim = (max_sh_degree + 1) ** 2

            if model_name == 'background':                  
                shs = model.get_features.transpose(1, 2).view(-1, 3, sh_dim)
            else:
                features = model.get_features_fourier(self.frame)
                shs = features.transpose(1, 2).view(-1, 3, sh_dim)

            directions = model.get_xyz - camera_center
            directions = directions / torch.norm(directions, dim=1, keepdim=True)
            sh2rgb = eval_sh(max_sh_degree, shs, directions)
            color = torch.clamp_min(sh2rgb + 0.5, 0.)
            colors.append(color)

        colors = torch.cat(colors, dim=0)
        return colors
                

    @property
    def get_semantic(self):
        semantics = []
        if self.get_visibility('background'):
            semantic_bkgd = self.background.get_semantic
            semantics.append(semantic_bkgd)

        for obj_name in self.graph_obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)
            
            semantic = obj_model.get_semantic
        
            semantics.append(semantic)

        semantics = torch.cat(semantics, dim=0)
        return semantics
    
    @property
    def get_opacity(self):
        opacities = []
        
        if self.get_visibility('background'):
            opacity_bkgd = self.background.get_opacity
            opacities.append(opacity_bkgd)

        for obj_name in self.graph_obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)
            
            opacity = obj_model.get_opacity
        
            opacities.append(opacity)
        
        opacities = torch.cat(opacities, dim=0)
        return opacities
            
    def get_covariance(self, scaling_modifier = 1):
        scaling = self.get_scoaling # [N, 1]
        rotation = self.get_rotation # [N, 4]
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm
    
    def get_normals(self, camera: Camera):
        normals = []
        
        if self.get_visibility('background'):
            normals_bkgd = self.background.get_normals(camera)            
            normals.append(normals_bkgd)
            
        for i, obj_name in enumerate(self.graph_obj_list):
            obj_model: GaussianModelActor = getattr(self, obj_name)
            track_id = obj_model.track_id

            normals_obj_local = obj_model.get_normals(camera) # [N, 3]
                    
            obj_rot = self.actor_pose.get_tracking_rotation(track_id, self.viewpoint_camera)
            obj_rot = quaternion_to_matrix(obj_rot.unsqueeze(0)).squeeze(0)
            
            normals_obj_global = normals_obj_local @ obj_rot.T
            normals_obj_global = torch.nn.functinal.normalize(normals_obj_global)                
            normals.append(normals_obj_global)

        normals = torch.cat(normals, dim=0)
        return normals
            
    def oneupSHdegree(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if model_name in exclude_list:
                continue
            model: GaussianModel = getattr(self, model_name)
            model.oneupSHdegree()
                    
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, exclude_list=[]):
        self.active_sh_degree = 0

        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.training_setup()
                
        if self.actor_pose is not None:
            self.actor_pose.training_setup()
        
        if self.sky_cubemap is not None:
            self.sky_cubemap.training_setup()
            
        if self.color_correction is not None:
            self.color_correction.training_setup()
            
        if self.pose_correction is not None:
            self.pose_correction.training_setup()
        
    def update_learning_rate(self, iteration, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.update_learning_rate(iteration)
        
        if self.actor_pose is not None:
            self.actor_pose.update_learning_rate(iteration)
    
        if self.sky_cubemap is not None:
            self.sky_cubemap.update_learning_rate(iteration)
            
        if self.color_correction is not None:
            self.color_correction.update_learning_rate(iteration)
            
        if self.pose_correction is not None:
            self.pose_correction.update_learning_rate(iteration)
    
    def update_optimizer(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.update_optimizer()

        if self.actor_pose is not None:
            self.actor_pose.update_optimizer()
        
        if self.sky_cubemap is not None:
            self.sky_cubemap.update_optimizer()
            
        if self.color_correction is not None:
            self.color_correction.update_optimizer()
            
        if self.pose_correction is not None:
            self.pose_correction.update_optimizer()

    def set_max_radii2D(self, radii, visibility_filter):
        radii = radii.float()
        
        for model_name in self.graph_gaussian_range.keys():
            model: GaussianModel = getattr(self, model_name)
            start, end = self.graph_gaussian_range[model_name]
            end += 1
            visibility_model = visibility_filter[start:end]
            max_radii2D_model = radii[start:end]
            model.max_radii2D[visibility_model] = torch.max(
                model.max_radii2D[visibility_model], max_radii2D_model[visibility_model])
        
    def add_densification_stats(self, viewspace_point_tensor, visibility_filter, pixels):
        viewspace_point_tensor_grad = viewspace_point_tensor.grad

        for model_name in self.graph_gaussian_range.keys():
            model: GaussianModel = getattr(self, model_name)
            start, end = self.graph_gaussian_range[model_name]
            end += 1
            visibility_model = visibility_filter[start:end]
            pixels_cp = pixels[start:end]
            viewspace_point_tensor_grad_model = viewspace_point_tensor_grad[start:end]
            model.xyz_gradient_accum[visibility_model, 0:1] += torch.norm(viewspace_point_tensor_grad_model[visibility_model, :2], dim=-1, keepdim=True)* pixels_cp[visibility_model]
            model.xyz_gradient_accum[visibility_model, 1:2] += torch.norm(viewspace_point_tensor_grad_model[visibility_model, 2:], dim=-1, keepdim=True)* pixels_cp[visibility_model]
            model.denom[visibility_model] += pixels_cp[visibility_model]
        
    def densify_and_prune(self, max_grad, min_opacity, prune_big_points, exclude_list=[]):
        scalars = None
        tensors = None
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)

            scalars_, tensors_ = model.densify_and_prune(max_grad, min_opacity, prune_big_points)
            if model_name == 'background':
                scalars = scalars_
                tensors = tensors_
    
        return scalars, tensors
    
    def get_box_reg_loss(self):
        box_reg_loss = 0.
        for obj_name in self.obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)
            box_reg_loss += obj_model.box_reg_loss()
        box_reg_loss /= len(self.obj_list)

        return box_reg_loss
            
    def reset_opacity(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            if startswith_any(model_name, exclude_list):
                continue
            model.reset_opacity()
