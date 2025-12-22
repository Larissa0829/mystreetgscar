import os
import torch
from random import randint
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim
from lib.utils.img_utils import save_img_torch, visualize_depth_numpy
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.gaussian_model_actor import GaussianModelActor
from lib.utils.general_utils import safe_state
from lib.utils.camera_utils import Camera
from lib.utils.cfg_utils import save_cfg
from lib.utils.lpipsPyTorch import lpips
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.config import cfg
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from lib.utils.system_utils import searchForMaxIteration
import time
import sys
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np


# 第四章 补全车辆 引入difix3d
# 添加 Difix3D/src 目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'Difix3D', 'src'))
from pipeline_difix import DifixPipeline
# 添加 TRELLIS目录到python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'TRELLIS'))


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False




def training():
    training_args = cfg.train
    optim_args = cfg.optim
    data_args = cfg.data

    start_iter = 0
    tb_writer = prepare_output_and_logger()
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)

    gaussians.training_setup()

    # ============ 创建difix3d
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data_args.isDifix:
        # 有两种，一种是不需要ref的
        difixPipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
        # 一种是需要ref的
        # difixPipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
        difixPipe.to(device)
        difixPrompt = "remove degradation"
    # ============


    try:
        if cfg.loaded_iter == -1:
            loaded_iter = searchForMaxIteration(cfg.trained_model_dir)
        else:
            loaded_iter = cfg.loaded_iter
        ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{loaded_iter}.pth')
        state_dict = torch.load(ckpt_path)
        start_iter = state_dict['iter']
        print(f'Loading model from {ckpt_path}')
        gaussians.load_state_dict(state_dict)
    except:
        pass

    print(f'Starting from {start_iter}')
    save_cfg(cfg, cfg.model_path, epoch=start_iter)

    gaussians_renderer = StreetGaussianRenderer()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_dict = {}
    progress_bar = tqdm(range(start_iter, training_args.iterations))
    start_iter += 1

    viewpoint_stack = None
    for iteration in range(start_iter, training_args.iterations + 1):
    
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Every 1000 iterations upsample
        # if iteration % 1000 == 0:
        #     if resolution_scales:  
        #         scale = resolution_scales.pop()


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    
        # ====================================================================
        # Get mask
        # original_mask: pixel in original_mask with 0 will not be surpervised
        # original_acc_mask: use to suepervise the acc result of rendering
        # original_sky_mask: sky mask

        gt_image = viewpoint_cam.original_image.cuda()
        if hasattr(viewpoint_cam, 'original_mask'):
            mask = viewpoint_cam.original_mask.cuda().bool()
        else:
            mask = torch.ones_like(gt_image[0:1]).bool()
        
        if hasattr(viewpoint_cam, 'original_sky_mask'):
            sky_mask = viewpoint_cam.original_sky_mask.cuda()
        else:
            sky_mask = None
            
        if hasattr(viewpoint_cam, 'original_obj_bound'):
            obj_bound = viewpoint_cam.original_obj_bound.cuda().bool()
        else:
            obj_bound = torch.zeros_like(gt_image[0:1]).bool()
        
        if (iteration - 1) == training_args.debug_from:
            cfg.render.debug = True
            
        render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians, depth_threshold = optim_args.depth_threshold * dataset.cameras_extent)
        image, acc, viewspace_point_tensor, visibility_filter, radii = render_pkg["rgb"], render_pkg['acc'], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth = render_pkg['depth'] # [1, H, W]

        scalar_dict = dict()
        # rgb loss
        Ll1 = l1_loss(image, gt_image, mask)
        scalar_dict['l1_loss'] = Ll1.item()
        loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + optim_args.lambda_dssim * (1.0 - ssim(image, gt_image, mask=mask))

        # sky loss
        if optim_args.lambda_sky > 0 and gaussians.include_sky and sky_mask is not None:
            acc = torch.clamp(acc, min=1e-6, max=1.-1e-6)
            sky_loss = torch.where(sky_mask, -torch.log(1 - acc), -torch.log(acc)).mean()
            if len(optim_args.lambda_sky_scale) > 0:
                sky_loss *= optim_args.lambda_sky_scale[viewpoint_cam.meta['cam']]
            scalar_dict['sky_loss'] = sky_loss.item()
            loss += optim_args.lambda_sky * sky_loss

        # semantic loss
        if optim_args.lambda_semantic > 0 and data_args.get('use_semantic', False) and 'semantic' in viewpoint_cam.meta:
            gt_semantic = viewpoint_cam.meta['semantic'].cuda().long() # [1, H, W]
            if torch.all(gt_semantic == -1):
                semantic_loss = torch.zeros_like(Ll1)
            else:
                semantic = render_pkg['semantic'].unsqueeze(0) # [1, S, H, W]
                semantic_loss = torch.nn.functional.cross_entropy(
                    input=semantic, 
                    target=gt_semantic,
                    ignore_index=-1, 
                    reduction='mean'
                )
            scalar_dict['semantic_loss'] = semantic_loss.item()
            loss += optim_args.lambda_semantic * semantic_loss
        
        if optim_args.lambda_reg > 0 and gaussians.include_obj and iteration >= optim_args.densify_until_iter:
            render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians)
            image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = torch.clamp(acc_obj, min=1e-6, max=1.-1e-6)

            # box_reg_loss = gaussians.get_box_reg_loss()
            # scalar_dict['box_reg_loss'] = box_reg_loss.item()
            # loss += optim_args.lambda_reg * box_reg_loss

            obj_acc_loss = torch.where(obj_bound, 
                -(acc_obj * torch.log(acc_obj) +  (1. - acc_obj) * torch.log(1. - acc_obj)), 
                -torch.log(1. - acc_obj)).mean()
            scalar_dict['obj_acc_loss'] = obj_acc_loss.item()
            loss += optim_args.lambda_reg * obj_acc_loss
            # obj_acc_loss = -((acc_obj * torch.log(acc_obj) +  (1. - acc_obj) * torch.log(1. - acc_obj))).mean()
            # scalar_dict['obj_acc_loss'] = obj_acc_loss.item()
            # loss += optim_args.lambda_reg * obj_acc_loss
        
        # lidar depth loss
        if optim_args.lambda_depth_lidar > 0 and 'lidar_depth' in viewpoint_cam.meta:            
            lidar_depth = viewpoint_cam.meta['lidar_depth'].cuda() # [1, H, W]
            depth_mask = torch.logical_and((lidar_depth > 0.), mask)
            # depth_mask[obj_bound] = False
            if torch.nonzero(depth_mask).any():
                expected_depth = depth / (render_pkg['acc'] + 1e-10)  
                depth_error = torch.abs((expected_depth[depth_mask] - lidar_depth[depth_mask]))
                depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)
                lidar_depth_loss = depth_error.mean()
                scalar_dict['lidar_depth_loss'] = lidar_depth_loss
            else:
                lidar_depth_loss = torch.zeros_like(Ll1)  
            loss += optim_args.lambda_depth_lidar * lidar_depth_loss
                    
        # color correction loss
        if optim_args.lambda_color_correction > 0 and gaussians.use_color_correction:
            color_correction_reg_loss = gaussians.color_correction.regularization_loss(viewpoint_cam)
            scalar_dict['color_correction_reg_loss'] = color_correction_reg_loss.item()
            loss += optim_args.lambda_color_correction * color_correction_reg_loss
        
        # pose correction loss
        if optim_args.lambda_pose_correction > 0 and gaussians.use_pose_correction:
            pose_correction_reg_loss = gaussians.pose_correction.regularization_loss()
            scalar_dict['pose_correction_reg_loss'] = pose_correction_reg_loss.item()
            loss += optim_args.lambda_pose_correction * pose_correction_reg_loss
                    
        # scale flatten loss
        if optim_args.lambda_scale_flatten > 0:
            scale_flatten_loss = gaussians.background.scale_flatten_loss()
            scalar_dict['scale_flatten_loss'] = scale_flatten_loss.item()
            loss += optim_args.lambda_scale_flatten * scale_flatten_loss
        
        # opacity sparse loss
        if optim_args.lambda_opacity_sparse > 0:
            opacity = gaussians.get_opacity
            opacity = opacity.clamp(1e-6, 1-1e-6)
            log_opacity = opacity * torch.log(opacity)
            log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
            sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[visibility_filter].mean()
            scalar_dict['opacity_sparse_loss'] = sparse_loss.item()
            loss += optim_args.lambda_opacity_sparse * sparse_loss
                
        # normal loss
        if optim_args.lambda_normal_mono > 0 and 'mono_normal' in viewpoint_cam.meta and 'normals' in render_pkg:
            if sky_mask is None:
                normal_mask = mask
            else:
                normal_mask = torch.logical_and(mask, ~sky_mask)
                normal_mask = normal_mask.squeeze(0)
                normal_mask[:50] = False
                
            normal_gt = viewpoint_cam.meta['mono_normal'].permute(1, 2, 0).cuda() # [H, W, 3]
            R_c2w = viewpoint_cam.world_view_transform[:3, :3]
            normal_gt = torch.matmul(normal_gt, R_c2w.T) # to world space
            normal_pred = render_pkg['normals'].permute(1, 2, 0) # [H, W, 3]    
            
            normal_l1_loss = torch.abs(normal_pred[normal_mask] - normal_gt[normal_mask]).mean()
            normal_cos_loss = (1. - torch.sum(normal_pred[normal_mask] * normal_gt[normal_mask], dim=-1)).mean()
            scalar_dict['normal_l1_loss'] = normal_l1_loss.item()
            scalar_dict['normal_cos_loss'] = normal_cos_loss.item()
            normal_loss = normal_l1_loss + normal_cos_loss
            loss += optim_args.lambda_normal_mono * normal_loss


        # difix3d loss: 目前想法是把difix3d嵌入 生成一个去除伪影的效果，来得到完整的车辆。（由于在渲染的时候会执行，所以训练的时候的优化有点微不足道）
        to_tensor = transforms.ToTensor()
        if data_args.isDifix and iteration % 2500 == 0:
            # rgb
            difix_rgb = to_tensor(difixPipe(difixPrompt, image=image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]).to(device)
            difix_rgb = F.interpolate(difix_rgb.unsqueeze(0), size=(1066, 1600), mode='bilinear', align_corners=False).squeeze(0)
            Ll1 = l1_loss(image, difix_rgb, mask)
            scalar_dict['difix3d_loss'] = Ll1.item()
            loss += (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + optim_args.lambda_dssim * (1.0 - ssim(image, difix_rgb, mask=mask))

        if data_args.isDifix and iteration > int(training_args.iterations * 2 // 3)+1 and iteration % 250 == 1:
            # obj + 随机移动或旋转
            if gaussians.include_obj:
                # 生成随机旋转和平移
                with torch.no_grad():
                    custom_rotation = torch.tensor([torch.randint(90, 91, (1,)).item()], dtype=torch.float32) * torch.pi / 180
                    custom_translation = torch.empty(1).uniform_(0, 0)
                
                # 构建包含原始对象和对应sample对象的列表
                origin_list = [name for name in gaussians.obj_list if not name.endswith('_sample') and name not in ['sky', 'background']]
                sample_list = []
                for obj_name in origin_list:
                    sample_name = f"{obj_name}_sample"
                    if hasattr(gaussians, sample_name) and sample_name in gaussians.model_name_id:
                        sample_list.append(sample_name)
                
                # 合并原始对象和sample对象进行渲染（sample点云补充原始对象）
                include_obj_list = origin_list + sample_list
                
                # 渲染：rgb_obj为随机变换之后的图像（包含原始对象+sample点云补充）
                # 注意：需要在有梯度的情况下渲染，以便后续loss可以反向传播优化gaussians参数
                rgb_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians, custom_rotation=custom_rotation, custom_translation=custom_translation, include_list=include_obj_list)["rgb"]
                
                # difix_rgb_obj为difix3d处理之后的图像，作为临时的gt
                # difixPipe推理可以放在no_grad下（因为它是固定模型，不需要梯度）
                with torch.no_grad():
                    # 使用difix3d方法：可以输入参考ref（如果需要，可以取消注释下面一行并注释当前行）
                    difix_rgb_obj = to_tensor(difixPipe(difixPrompt, image=rgb_obj.detach(), num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]).to(device)
                    # ref_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians, custom_rotation=None, custom_translation=None, include_list=origin_list)["rgb"]
                    # difix_rgb_obj = to_tensor(difixPipe(difixPrompt, image=rgb_obj.detach(), ref_image=ref_obj.detach(), num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]).to(device)
                    difix_rgb_obj = F.interpolate(difix_rgb_obj.unsqueeze(0), size=(1066, 1600), mode='bilinear', align_corners=False).squeeze(0)
                    
                    # 存储车辆部分的图像，方便查看
                    transform = T.ToPILImage()
                    image_pil = transform(rgb_obj.detach().cpu().clamp(0, 1))  # 确保 image 在 CPU 上
                    image_pil.save(f"rgb_obj_{iteration}_wodifix.png")
                    image_pil = transform(difix_rgb_obj.detach().cpu().clamp(0, 1))  # 确保 image 在 CPU 上
                    image_pil.save(f"rgb_obj_{iteration}_wdifix.png")
                
                # 计算loss（需要在有梯度的情况下计算，以便反向传播优化gaussians参数）
                Ll1 = l1_loss(rgb_obj, difix_rgb_obj, mask)
                scalar_dict['difix3d_obj_loss'] = Ll1.item()
                loss += (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + optim_args.lambda_dssim * (1.0 - ssim(rgb_obj, difix_rgb_obj, mask=mask))


        # ================================ TRELLIS 单帧点云模板生成（作为独立对象）================================
        if iteration == int(training_args.iterations * 2 // 3)+1 and gaussians.include_obj and data_args.isTrellis:
            os.environ['SPCONV_ALGO'] = 'native'
            from PIL import Image
            from trellis.pipelines import TrellisImageTo3DPipeline
            from lib.utils.general_utils import matrix_to_quaternion, quaternion_raw_multiply
            
            print("=" * 80)
            print("TRELLIS 点云模板生成与对齐")
            print("=" * 80)
            
            ply_file_path = f"{cfg.model_path}/input_ply"
            os.makedirs(ply_file_path, exist_ok=True)
            
            # ============ 阶段1: 生成点云 ============
            print("\n[阶段1] 生成TRELLIS点云模板...")
            trellis_pipeline = None
            trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
            trellis_pipeline.to(device)
            
            for obj_name in gaussians.obj_list:
                # 跳过背景、天空和已经存在的 sample 对象
                if obj_name in ['sky', 'background'] or obj_name.endswith('_sample'):
                    continue
                
                sample_ply = f"{ply_file_path}/{obj_name}_sample.ply"
                if os.path.exists(sample_ply):
                    print(f"  ✓ {obj_name}_sample.ply 已存在，跳过生成")
                    continue
                
                print(f"\n  生成 {obj_name} 的点云...")
                
                try:
                    # 渲染对象图像
                    with torch.no_grad():
                        render_obj = gaussians_renderer.render_object(
                            viewpoint_cam, gaussians, 
                            custom_rotation=None, custom_translation=None,
                            include_list=[obj_name]
                        )
                        rgb_obj = torch.clamp(render_obj["rgb"], 0, 1).permute(1, 2, 0)
                        alpha = torch.clamp(render_obj['acc'], 0, 1).permute(1, 2, 0)
                        rgba = torch.cat([rgb_obj, alpha], dim=-1)
                        image_pil = Image.fromarray((rgba * 255).byte().cpu().numpy(), mode="RGBA")
                        image_pil.save(f"{ply_file_path}/{obj_name}_input.png")
                    
                    # 运行TRELLIS生成点云
                    outputs = trellis_pipeline.run(
                        image_pil, seed=1, formats=['gaussian'],
                        sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
                        slat_sampler_params={"steps": 12, "cfg_strength": 3}
                    )
                    obj_trellis_ply = outputs['gaussian'][0]
                    
                    # 添加高阶球谐系数
                    if obj_trellis_ply._features_rest is None:
                        num_points = obj_trellis_ply._features_dc.shape[0]
                        obj_trellis_ply._features_rest = torch.zeros(
                            (num_points, 15, 3),  # sh_degree=3: (3+1)^2-1=15
                            dtype=obj_trellis_ply._features_dc.dtype,
                            device=obj_trellis_ply._features_dc.device
                        )
                        obj_trellis_ply.sh_degree = obj_trellis_ply.active_sh_degree = 3
                    
                    obj_trellis_ply.save_ply(sample_ply)
                    print(f"    ✓ 已保存: {sample_ply} ({num_points} 点)")
                    
                except Exception as e:
                    print(f"    ⚠️ TRELLIS生成失败: {str(e)}")
                    print(f"    跳过 {obj_name}，继续处理下一个对象")
                    continue
                
                torch.cuda.empty_cache()
            
            if trellis_pipeline is not None:
                del trellis_pipeline

            torch.cuda.empty_cache()
            

            # ============ 阶段2: 创建并对齐sample对象 ============
            print("\n[阶段2] 创建独立的sample对象并对齐...")
            
            # 坐标系转换矩阵（TRELLIS坐标系 → 目标坐标系）
            transform_matrix = torch.tensor([
                [0,  0, -1],  # X: -Z
                [1,  0,  0],  # Y: +X
                [0, -1,  0]   # Z: -Y
            ], device='cuda', dtype=torch.float32)
            transform_quat = matrix_to_quaternion(transform_matrix.unsqueeze(0)).squeeze(0)
            
            obj_list = gaussians.obj_list.copy()  # 避免迭代时修改列表
            for obj_name in obj_list:
                # 跳过背景、天空和已经存在的 sample 对象
                if obj_name in ['sky', 'background'] or obj_name.endswith('_sample'):
                    continue
                
                sample_name = f"{obj_name}_sample"
                sample_ply = f"{ply_file_path}/{sample_name}.ply"
                if not os.path.exists(sample_ply):
                    continue
                
                actor: GaussianModelActor = getattr(gaussians, obj_name)
                
                # 检查 sample 对象是否已经注册（从 checkpoint 加载）
                if hasattr(gaussians, sample_name) and sample_name in gaussians.model_name_id:
                    print(f"\n  {sample_name} 已存在（从 checkpoint），重新对齐...")
                    sample_actor = getattr(gaussians, sample_name)
                    
                    # 【关键修复】从原始 PLY 文件重新加载颜色特征（修复 checkpoint 中可能被破坏的颜色）
                    print(f"    从原始 PLY 文件重新加载颜色特征...")
                    from plyfile import PlyData
                    plydata = PlyData.read(sample_ply)
                    
                    # 读取基础颜色特征 (DC component)
                    features_dc = np.zeros((len(plydata['vertex']), 3, 1))
                    features_dc[:, 0, 0] = np.asarray(plydata['vertex']['f_dc_0'])
                    features_dc[:, 1, 0] = np.asarray(plydata['vertex']['f_dc_1'])
                    features_dc[:, 2, 0] = np.asarray(plydata['vertex']['f_dc_2'])
                    
                    # 读取 rest 特征
                    extra_f_names = [p.name for p in plydata['vertex'].properties if p.name.startswith("f_rest_")]
                    if extra_f_names:
                        features_rest = np.zeros((len(plydata['vertex']), len(extra_f_names)))
                        for idx, attr_name in enumerate(extra_f_names):
                            features_rest[:, idx] = np.asarray(plydata['vertex'][attr_name])
                        features_rest = features_rest.reshape((len(plydata['vertex']), 3, -1))
                    else:
                        features_rest = np.zeros((len(plydata['vertex']), 3, 0))
                    
                    # 调整 sh_degree 维度以匹配原始对象
                    target_sh_degree = actor.max_sh_degree
                    ply_sh_rest_dim = features_rest.shape[2]
                    target_sh_rest_dim = (target_sh_degree + 1) ** 2 - 1
                    
                    if ply_sh_rest_dim != target_sh_rest_dim:
                        features_rest_new = np.zeros((len(plydata['vertex']), 3, target_sh_rest_dim))
                        copy_dim = min(ply_sh_rest_dim, target_sh_rest_dim)
                        features_rest_new[:, :, :copy_dim] = features_rest[:, :, :copy_dim]
                        features_rest = features_rest_new
                    
                    # 转换为 torch 张量并调整维度
                    features_dc_tensor = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
                    features_rest_tensor = torch.tensor(features_rest, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
                    
                    # 调整 fourier_dim（只影响 _features_dc）
                    target_fourier_dim = actor._features_dc.shape[1]
                    if target_fourier_dim > 1:
                        # 【关键修复】只在第一个 fourier 通道放值，其他通道放0
                        # 原因：IDFT 权重的和不是1，如果复制到所有通道会导致颜色被放大
                        num_points = features_dc_tensor.shape[0]
                        features_dc_expanded = torch.zeros((num_points, target_fourier_dim, 3), device='cuda', dtype=torch.float)
                        features_dc_expanded[:, 0, :] = features_dc_tensor[:, 0, :]  # 只在第一个通道放值
                        features_dc_tensor = features_dc_expanded
                        print(f"    扩展 fourier_dim: 1 -> {target_fourier_dim} (只在第0通道放值)")
                    
                    # 更新 sample_actor 的颜色特征（暂存到 data 属性，后面会转为 Parameter）
                    sample_actor._features_dc_data = features_dc_tensor
                    sample_actor._features_rest_data = features_rest_tensor
                    
                    print(f"    ✓ 颜色特征已从 PLY 恢复: dc_sum={features_dc_tensor.abs().sum().item():.2f}, rest_sum={features_rest_tensor.abs().sum().item():.2f}")
                    
                    need_register = False
                else:
                    print(f"\n  创建 {sample_name}...")
                    # 加载点云
                    sample_actor = GaussianModelActor(model_name=sample_name, obj_meta=actor.obj_meta)
                    sample_actor.load_ply(sample_ply)
                    print(f"    点数: {sample_actor._xyz.shape[0]}")
                    
                    # 标记为新创建，后面会使用 load_ply 加载的颜色
                    sample_actor._features_dc_data = None
                    sample_actor._features_rest_data = None
                    
                    need_register = True
                
                with torch.no_grad():
                    # 检查原始对象是否有点云
                    obj_xyz = actor._xyz.data
                    if obj_xyz.shape[0] == 0:
                        print(f"    ⚠️  警告：原始对象 {obj_name} 没有点云（可能被完全修剪），跳过 sample 对齐")
                        continue
                    
                    # 检查 sample 对象是否有点云
                    template_xyz = sample_actor._xyz.data.cuda()
                    if template_xyz.shape[0] == 0:
                        print(f"    ⚠️  警告：sample 对象 {sample_name} 没有点云，跳过对齐")
                        continue
                    
                    # 1. 坐标转换与缩放
                    t_center = (template_xyz.min(dim=0)[0] + template_xyz.max(dim=0)[0]) / 2
                    template_xyz_transformed = torch.matmul(template_xyz - t_center, transform_matrix.T)
                    
                    # 计算缩放因子（基于最大维度）
                    t_scale = template_xyz_transformed.max(dim=0)[0] - template_xyz_transformed.min(dim=0)[0]
                    obj_scale = obj_xyz.max(dim=0)[0] - obj_xyz.min(dim=0)[0]
                    scale_factor = obj_scale.max() / (t_scale.max() + 1e-8)
                    
                    template_xyz_scaled = template_xyz_transformed * scale_factor
                    print(f"    缩放因子: {scale_factor.item():.4f}")
                    
                    # 2. 更新所有参数
                    sample_actor._xyz = torch.nn.Parameter(template_xyz_scaled.requires_grad_(True))
                    sample_actor._rotation = torch.nn.Parameter(quaternion_raw_multiply(transform_quat.unsqueeze(0).expand(sample_actor._rotation.shape[0], -1),sample_actor._rotation.data.cuda()).requires_grad_(True))
                    sample_actor._scaling = torch.nn.Parameter((sample_actor._scaling.data.cuda() + torch.log(scale_factor)).requires_grad_(True))
                    sample_actor._opacity = torch.nn.Parameter(sample_actor._opacity.data.cuda().requires_grad_(True))
                    sample_actor._semantic = torch.nn.Parameter(sample_actor._semantic.data.cuda().requires_grad_(True))
                    
                    # 3. 调整特征维度（使用从 PLY 重新加载的颜色特征）
                    if hasattr(sample_actor, '_features_dc_data') and sample_actor._features_dc_data is not None:
                        # 使用从 PLY 文件重新加载的颜色特征（checkpoint 中已存在的情况）
                        sample_actor._features_dc = torch.nn.Parameter(sample_actor._features_dc_data.requires_grad_(True))
                        sample_actor._features_rest = torch.nn.Parameter(sample_actor._features_rest_data.requires_grad_(True))
                        print(f"    ✓ 使用从 PLY 重新加载的颜色特征")
                    else:
                        # 新创建的 sample，使用 load_ply 加载的颜色（需要调整 fourier_dim）
                        sample_dc = sample_actor._features_dc.data.cuda()
                        sample_fourier_dim = sample_dc.shape[1]
                        actor_fourier_dim = actor._features_dc.shape[1]
                        
                        if sample_fourier_dim != actor_fourier_dim:
                            print(f"    调整 fourier_dim: {sample_fourier_dim} -> {actor_fourier_dim}")
                            if sample_fourier_dim < actor_fourier_dim:
                                # 【关键修复】只在第一个 fourier 通道放值，其他通道放0
                                num_points = sample_dc.shape[0]
                                sample_dc_new = torch.zeros((num_points, actor_fourier_dim, 3), device='cuda', dtype=torch.float)
                                sample_dc_new[:, 0, :] = sample_dc[:, 0, :]
                                sample_dc = sample_dc_new
                                print(f"      (只在第0通道放值，其他通道为0)")
                            else:
                                sample_dc = sample_dc[:, :actor_fourier_dim, :]
                        
                        sample_actor._features_dc = torch.nn.Parameter(sample_dc.requires_grad_(True))
                        
                        # _features_rest：保留原始信息
                        if sample_actor._features_rest is not None:
                            sample_rest = sample_actor._features_rest.data.cuda()
                            sample_sh_rest_dim = sample_rest.shape[1]
                            actor_sh_rest_dim = actor._features_rest.shape[1]
                            
                            if sample_sh_rest_dim != actor_sh_rest_dim:
                                num_points = sample_actor._xyz.shape[0]
                                sample_rest_new = torch.zeros((num_points, actor_sh_rest_dim, 3), device='cuda')
                                copy_dim = min(sample_sh_rest_dim, actor_sh_rest_dim)
                                sample_rest_new[:, :copy_dim, :] = sample_rest[:, :copy_dim, :]
                                sample_rest = sample_rest_new
                            
                            sample_actor._features_rest = torch.nn.Parameter(sample_rest.requires_grad_(True))
                        else:
                            sample_actor._features_rest = torch.nn.Parameter(
                                torch.zeros((sample_actor._xyz.shape[0], actor._features_rest.shape[1], 3), device='cuda').requires_grad_(True)
                            )
                        
                        print(f"    ✓ 使用 load_ply 加载的颜色特征（已调整维度）")
                    
                    sample_actor.max_sh_degree = actor.max_sh_degree
                    sample_actor.active_sh_degree = actor.active_sh_degree
                
                # 3.5. 【必须】设置时间戳，否则 parse_camera 会跳过 sample
                sample_actor.track_id = actor.track_id
                sample_actor.start_timestamp = actor.start_timestamp
                sample_actor.end_timestamp = actor.end_timestamp
                
                # 4. 【必须】重新初始化 optimizer（因为参数对象变了）
                # 否则 optimizer 会指向旧的参数，导致颜色不更新
                sample_actor.training_setup()
                # 注意：维度必须正确！xyz_gradient_accum: [N, 2], denom: [N, 1], max_radii2D: [N]
                sample_actor.max_radii2D = torch.zeros(sample_actor._xyz.shape[0], dtype=torch.float32, device='cuda')
                sample_actor.xyz_gradient_accum = torch.zeros((sample_actor._xyz.shape[0], 2), dtype=torch.float32, device='cuda')
                sample_actor.denom = torch.zeros((sample_actor._xyz.shape[0], 1), dtype=torch.float32, device='cuda')
                
                # 5. 注册到场景（仅新创建的）
                if need_register:
                    setattr(gaussians, sample_name, sample_actor)
                    gaussians.model_name_id[sample_name] = gaussians.models_num
                    gaussians.obj_list.append(sample_name)
                    gaussians.models_num += 1
                    print(f"    ✓ 已注册，track_id={actor.track_id}（跟随{obj_name}）")
                else:
                    print(f"    ✓ 已重新对齐，track_id={actor.track_id}（跟随{obj_name}）")
            
            print("\n" + "=" * 80)
            print(f"✓ 完成! 总对象数: {gaussians.models_num}")
            print("=" * 80)
            torch.cuda.empty_cache()
            
            # 测试sample是否创建成功
            sample = gaussians_renderer.render_object(viewpoint_cam, gaussians,custom_rotation=None,custom_translation=None,include_list=["obj_004_sample"])["rgb"]
            transform = T.ToPILImage()
            sample = transform(sample.cpu())  # 确保 image 在 CPU 上
            sample.save("sample.png")
            
            # # ============ 阶段3: 对原始动态对象应用对称性补全 ============
            # print("\n[阶段3] 对原始动态对象应用自适应对称性补全...")
            # from lib.utils.symmetry_utils import apply_symmetry_to_all_objects
            
            # # 对原始对象（非sample）应用自适应镜像补全
            # # 每辆车自动判断哪一侧观测充分，然后镜像到另一侧
            # apply_symmetry_to_all_objects(
            #     gaussians, 
            #     axis=1,  # Y轴对称（左右对称）
            #     mirror=True,  # 进行自适应镜像补全
            #     add_loss=False  # 此时不计算损失
            # )
            
            # print("  ✓ 原始对象自适应对称性补全完成")
            # print("=" * 80)
            # torch.cuda.empty_cache()
            
            # 【必须】重新解析场景图，否则 densification 会因为 graph_gaussian_range 不包含 sample 而报错
            print("\n重新解析场景图...")
            gaussians.set_visibility(include_list=list(set(gaussians.model_name_id.keys()) - set(['sky'])))
            gaussians.parse_camera(viewpoint_cam)
            print("  ✓ 场景图已更新，sample 对象已包含")
            torch.cuda.empty_cache()

            
            # 不 continue，让训练正常继续
        # ================================
        
        # # 对称性损失（从 16002 迭代开始，只对 sample 对象）
        # # 优化：每 5 次迭代计算一次，保证效果
        # if iteration > 16001 and iteration % 5 == 0 and gaussians.include_obj and data_args.isTrellis:
        #     from lib.utils.symmetry_utils import apply_symmetry_to_sample_objects
        #     symmetry_loss = apply_symmetry_to_sample_objects(
        #         gaussians, 
        #         axis=1,  # Y轴对称
        #         mirror=False,  # 不镜像，只计算损失
        #         add_loss=True  # 计算对称性损失
        #     )
        #     if symmetry_loss is not None and symmetry_loss > 0:
        #         scalar_dict['symmetry_loss'] = symmetry_loss.item()
        #         loss += symmetry_loss
            
        scalar_dict['loss'] = loss.item()

        loss.backward()
        
        iter_end.record()
                
        is_save_images = True
        if is_save_images and (iteration % 1000 == 1):
            # row0: gt_image, image, depth
            # row1: acc, image_obj, acc_obj
            depth_colored, _ = visualize_depth_numpy(depth.detach().cpu().numpy().squeeze(0))
            depth_colored = depth_colored[..., [2, 1, 0]] / 255.
            depth_colored = torch.from_numpy(depth_colored).permute(2, 0, 1).float().cuda()
            row0 = torch.cat([gt_image, image, depth_colored], dim=2)
            acc = acc.repeat(3, 1, 1)
            with torch.no_grad():
                render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians)
                image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = acc_obj.repeat(3, 1, 1)
            row1 = torch.cat([acc, image_obj, acc_obj], dim=2)
            with torch.no_grad():
                obj_name = gaussians.obj_list[0]
                origin_list = [name for name in gaussians.model_name_id.keys() if not name.endswith('_sample') and not name == 'background']
                sample_list = [name for name in gaussians.model_name_id.keys() if name.endswith('_sample')]

                image_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians,exclude_list=[],include_list=origin_list,custom_rotation=90)["rgb"]
                render_obj_sample = gaussians_renderer.render_object(viewpoint_cam, gaussians,exclude_list=[],include_list=sample_list,custom_rotation=90)
                image_obj_sample, acc_obj_sample = render_obj_sample["rgb"],render_obj_sample["acc"]
            acc_obj_sample = acc_obj_sample.repeat(3, 1, 1)
            row2 = torch.cat([image_obj, image_obj_sample, acc_obj_sample], dim=2) # 原始车 + sample + sample_acc

            image_to_show = torch.cat([row0, row1, row2], dim=1)
            image_to_show = torch.clamp(image_to_show, 0.0, 1.0)
            os.makedirs(f"{cfg.model_path}/log_images", exist_ok = True)
            save_img_torch(image_to_show, f"{cfg.model_path}/log_images/{iteration}.jpg")
        
        with torch.no_grad():
            # Log
            tensor_dict = dict()

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr(image, gt_image, mask).mean().float() + 0.6 * ema_psnr_for_log
            if viewpoint_cam.id not in psnr_dict:
                psnr_dict[viewpoint_cam.id] = psnr(image, gt_image, mask).mean().float()
            else:
                psnr_dict[viewpoint_cam.id] = 0.4 * psnr(image, gt_image, mask).mean().float() + 0.6 * psnr_dict[viewpoint_cam.id]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Exp": f"{cfg.task}-{cfg.exp_name}", 
                                          "Loss": f"{ema_loss_for_log:.{7}f},", 
                                          "PSNR": f"{ema_psnr_for_log:.{4}f}"})
                progress_bar.update(10)
            if iteration == training_args.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in training_args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < optim_args.densify_until_iter:
                # 【显存优化】排除sample点云，它们不参与densification相关操作
                include_list = [name for name in gaussians.model_name_id.keys() 
                               if not name.endswith('_sample') and name != 'sky']
                gaussians.set_visibility(include_list=include_list)
                gaussians.parse_camera(viewpoint_cam)   
                gaussians.set_max_radii2D(radii, visibility_filter)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, render_pkg["pixels"])
                
                prune_big_points = iteration > optim_args.opacity_reset_interval

                if iteration > optim_args.densify_from_iter:
                    if iteration % optim_args.densification_interval == 0:
                        # sample点云参与densify和prune，但使用更严格的阈值（在gaussian_model_actor中控制）
                        exclude_list = []  # 不再排除sample点云，让它们参与densify和prune
                        scalars, tensors = gaussians.densify_and_prune(
                            max_grad=optim_args.densify_grad_threshold,
                            min_opacity=optim_args.min_opacity,
                            prune_big_points=prune_big_points,
                            exclude_list=exclude_list,
                        )

                        scalar_dict.update(scalars)
                        tensor_dict.update(tensors)
                        
            # Reset opacity
            if iteration < optim_args.densify_until_iter:
                if iteration % optim_args.opacity_reset_interval == 0:
                    # 排除sample点云、sky和background，避免重置它们的透明度
                    exclude_list = ['_sample']
                    gaussians.reset_opacity(exclude_list=exclude_list)
                if data_args.white_background and iteration == optim_args.densify_from_iter:
                    exclude_list = ['_sample']
                    gaussians.reset_opacity(exclude_list=exclude_list)

            training_report(tb_writer, iteration, scalar_dict, tensor_dict, training_args.test_iterations, scene, gaussians_renderer, optim_args)

            # Optimizer step
            if iteration < training_args.iterations:
                gaussians.update_optimizer()

            if (iteration in training_args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                state_dict = gaussians.save_state_dict(is_final=(iteration == training_args.iterations))
                state_dict['iter'] = iteration
                ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{iteration}.pth')
                torch.save(state_dict, ckpt_path)




def prepare_output_and_logger():
    
    # if cfg.model_path == '':
    #     if os.getenv('OAR_JOB_ID'):
    #         unique_str = os.getenv('OAR_JOB_ID')
    #     else:
    #         unique_str = str(uuid.uuid4())
    #     cfg.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(cfg.model_path))

    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(cfg.trained_model_dir, exist_ok=True)
    os.makedirs(cfg.record_dir, exist_ok=True)
    if not cfg.resume:
        os.system('rm -rf {}/*'.format(cfg.record_dir))
        os.system('rm -rf {}/*'.format(cfg.trained_model_dir))

    with open(os.path.join(cfg.model_path, "cfg_args"), 'w') as cfg_log_f:
        viewer_arg = dict()
        viewer_arg['sh_degree'] = cfg.model.gaussian.sh_degree
        viewer_arg['white_background'] = cfg.data.white_background
        viewer_arg['source_path'] = cfg.source_path
        viewer_arg['model_path']= cfg.model_path
        cfg_log_f.write(str(Namespace(**viewer_arg)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(cfg.record_dir)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, scalar_stats, tensor_stats, testing_iterations, scene: Scene, renderer: StreetGaussianRenderer, optim_args):
    if tb_writer:
        try:
            for key, value in scalar_stats.items():
                tb_writer.add_scalar('train/' + key, value, iteration)
            for key, value in tensor_stats.items():
                tb_writer.add_histogram('train/' + key, value, iteration)
        except:
            print('Failed to write to tensorboard')
            
            
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test/test_view', 'cameras' : scene.getTestCameras()},
                              {'name': 'train/train_view', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderer.render(viewpoint, scene.gaussians, depth_threshold = optim_args.depth_threshold * scene.dataset.cameras_extent)["rgb"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    if hasattr(viewpoint, 'original_mask'):
                        mask = viewpoint.original_mask.cuda().bool()
                    else:
                        mask = torch.ones_like(gt_image[0]).bool()
                    l1_test += l1_loss(image, gt_image, mask).mean().double()
                    psnr_test += psnr(image, gt_image, mask).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("test/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('test/points_total', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Optimizing " + cfg.model_path)

    # Initialize system state (RNG)
    safe_state(cfg.train.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)
    training()

    # All done
    print("\nTraining complete.")