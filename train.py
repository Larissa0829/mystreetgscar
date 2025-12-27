import os
import torch
from random import randint
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim
from lib.utils.img_utils import save_img_torch, visualize_depth_numpy, draw_3d_box_on_img
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.gaussian_model_actor import GaussianModelActor
from lib.utils.general_utils import safe_state, quaternion_to_matrix, draw_bbox_for_objects
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
import cv2


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
        # 使用 float16 并开启 CPU 卸载以节省显存
        difixPipe = DifixPipeline.from_pretrained("nvidia/difix", torch_dtype=torch.float16, trust_remote_code=True)
        difixPipe.enable_model_cpu_offload() 
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


        # difix3d loss: 利用 Difix3D 优化未观测视角纹理，并结合 TRELLIS 几何先验进行强约束
        to_tensor = transforms.ToTensor()
        if data_args.isDifix and iteration % 2500 == 0:
            difix_rgb = to_tensor(difixPipe(difixPrompt, image=image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]).to(device)
            difix_rgb = F.interpolate(difix_rgb.unsqueeze(0), size=image.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
            Ll1 = l1_loss(image, difix_rgb, mask)
            loss += (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + optim_args.lambda_dssim * (1.0 - ssim(image, difix_rgb, mask=mask))

        if data_args.isDifix and iteration % 250 == 1:
            if gaussians.include_obj:
                # 1. 扩大随机旋转范围到 0-360 度，模拟全视角监督
                with torch.no_grad():
                    custom_rotation = torch.tensor([torch.randint(0, 360, (1,)).item()], dtype=torch.float32) * torch.pi / 180
                    custom_translation = torch.zeros(1).to(device)
                
                # 2. 准备对象列表 (原始点云 + TRELLIS补全点云)
                origin_list = [name for name in gaussians.obj_list if not name.endswith('_sample') and name not in ['sky', 'background']]
                sample_list = [name + "_sample" for name in origin_list if hasattr(gaussians, name + "_sample")]
                
                # 3. 多次渲染定位“观测真空区”
                with torch.no_grad():
                    # 仅渲染原始观察到的点云
                    render_origin = gaussians_renderer.render_object(viewpoint_cam, gaussians, custom_rotation=custom_rotation, include_list=origin_list)
                    origin_acc = render_origin["acc"]
                    
                # 渲染完整补全后的点云
                render_full = gaussians_renderer.render_object(viewpoint_cam, gaussians, custom_rotation=custom_rotation, include_list=origin_list + sample_list)
                rgb_obj = render_full["rgb"]
                full_acc = render_full["acc"]
                
                # 4. 识别 unobserved_mask: 只有补全点云填充了，但原始观察没拍到的地方
                unobserved_mask = (full_acc > 0.1) & (origin_acc < 0.3)
                obj_mask = (full_acc > 0.1)

                # 5. 利用 Difix3D 生成纹理伪标签 (低分辨率处理以节省显存)
                with torch.no_grad():
                    # 推理前清空缓存
                    torch.cuda.empty_cache()
                    
                    # 将渲染图下采样到 512 左右进行扩散模型推理，并使用 half 精度
                    rgb_obj_low = F.interpolate(rgb_obj.detach().unsqueeze(0), size=(768, 1152), mode='bilinear', align_corners=False).squeeze(0).half()
                    
                    difix_res = difixPipe(difixPrompt, image=rgb_obj_low, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
                    
                    # 转换回 float32 并插值回原始分辨率用于 Loss
                    difix_rgb_obj = to_tensor(difix_res).to(device).float()
                    difix_rgb_obj = F.interpolate(difix_rgb_obj.unsqueeze(0), size=rgb_obj.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
                    
                    # 立即清理中间变量
                    del rgb_obj_low, difix_res
                    torch.cuda.empty_cache()
                    
                    if iteration % 500 == 1:
                        save_dir = f"{cfg.model_path}/rotation_recording"
                        os.makedirs(save_dir, exist_ok=True)
                        T.ToPILImage()(rgb_obj.detach().cpu().clamp(0, 1)).save(os.path.join(save_dir, f"rgb_obj_{iteration}_wodifix.png"))
                        T.ToPILImage()(difix_rgb_obj.detach().cpu().clamp(0, 1)).save(os.path.join(save_dir, f"rgb_obj_{iteration}_wdifix.png"))
                
                # 6. 计算 RGB 损失：在观测不到的“蓝色空洞”区赋予极高权重，强制 sample 点云存活并着色
                loss_weight = torch.ones_like(full_acc)
                loss_weight[unobserved_mask] = 5.0 # 高权重，防止 sample 被优化器杀死
                
                Ll1_difix = (torch.abs(rgb_obj - difix_rgb_obj) * loss_weight * obj_mask).mean()
                scalar_dict['difix3d_obj_loss'] = Ll1_difix.item()
                loss += Ll1_difix * optim_args.lambda_l1

        # 7. TRELLIS 几何/不透明度强约束 (核心解决点云消失问题)
        if iteration > 500 and gaussians.include_obj:
            # A. 几何锚点损失 (Anchor Loss): 适应 Densification 的 Chamfer 距离版本
            if iteration % 10 == 0:
                anchor_loss = 0
                sample_count = 0
                for name in gaussians.model_name_id.keys():
                    if name.endswith('_sample'):
                        actor = getattr(gaussians, name)
                        if not hasattr(actor, 'initial_xyz'):
                            actor.initial_xyz = actor.get_xyz.detach().clone()
                        
                        curr_xyz = actor.get_xyz
                        ref_xyz = actor.initial_xyz
                        # 处理点数不匹配问题
                        if curr_xyz.shape[0] == ref_xyz.shape[0]:
                            dist = torch.norm(curr_xyz - ref_xyz, dim=-1)
                        else:
                            # 采样计算最近邻距离，保住几何轮廓
                            idx = torch.randperm(curr_xyz.shape[0])[:min(8000, curr_xyz.shape[0])]
                            tile = curr_xyz[idx]
                            dist_sq = torch.sum(tile**2, dim=1, keepdim=True) + torch.sum(ref_xyz**2, dim=1) - 2 * torch.matmul(tile, ref_xyz.t())
                            dist = torch.sqrt(torch.clamp(dist_sq.min(dim=1)[0], min=1e-7))
                        
                        anchor_loss += dist.mean()
                        sample_count += 1
                
                if sample_count > 0:
                    loss += (anchor_loss / sample_count) * 0.1 # 强形状约束
                    scalar_dict['anchor_loss'] = (anchor_loss / sample_count).item()

            # B. 不透明度保护 (Opacity Protection): 强制补全点云存活，不被 Prune 掉
            if iteration % 50 == 0:
                opacity_prot_loss = 0
                for name in gaussians.model_name_id.keys():
                    if name.endswith('_sample'):
                        actor = getattr(gaussians, name)
                        # 强制 sample 高斯维持最小透明度 0.5
                        opacity_prot_loss += torch.clamp(0.5 - actor.get_opacity, min=0).mean()
                if opacity_prot_loss > 0:
                    loss += opacity_prot_loss * 0.05
                    scalar_dict['opacity_prot_loss'] = opacity_prot_loss.item()

            # C. 对称性损失 (Symmetry Loss)
            from lib.utils.symmetry_utils import apply_symmetry_to_sample_objects
            symmetry_loss = apply_symmetry_to_sample_objects(gaussians, axis=1, mirror=False, add_loss=True)
            if symmetry_loss is not None and symmetry_loss > 0:
                sym_weight = 0.2 if iteration > 3000 else 0.05
                loss += symmetry_loss * sym_weight
                scalar_dict['symmetry_loss'] = symmetry_loss.item()

            # D. 缩放正则化: 防止高斯点无限膨胀
            if iteration % 20 == 0:
                scale_reg = 0
                for name in gaussians.model_name_id.keys():
                    if name.endswith('_sample'):
                        actor = getattr(gaussians, name)
                        scale_reg += torch.exp(actor.get_scaling).mean()
                if scale_reg > 0:
                    loss += scale_reg * 0.001
                    scalar_dict['scale_reg_loss'] = scale_reg.item()
        

            
        scalar_dict['loss'] = loss.item()

        loss.backward()
        
        iter_end.record()
                
        is_save_images = True
        if is_save_images and (iteration % 500 == 1):
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
                render_obj_sample_90 = gaussians_renderer.render_object(viewpoint_cam, gaussians,exclude_list=[],include_list=sample_list,custom_rotation=90)["rgb"]
                render_obj_sample_0 = gaussians_renderer.render_object(viewpoint_cam, gaussians,exclude_list=[],include_list=sample_list,custom_rotation=0)["rgb"]
                
                # # 在render_obj_sample_0上绘制原始车辆的bounding box
                # # 需要重新设置visibility，让原始对象也可见，这样才能在graph_obj_list中找到它们
                # gaussians.set_visibility(origin_list + sample_list)
                # gaussians.parse_camera(viewpoint_cam, custom_rotation=None, custom_translation=None)
                
                # render_obj_sample_0 = draw_bbox_for_objects(gaussians,viewpoint_cam,render_obj_sample_0,origin_list)

            row2 = torch.cat([image_obj, render_obj_sample_90, render_obj_sample_0], dim=2) # 原始车_90 + render_obj_sample_90 + render_obj_sample_0

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
                gaussians.set_visibility(include_list=list(set(gaussians.model_name_id.keys()) - set(['sky'])))
                gaussians.parse_camera(viewpoint_cam)   
                gaussians.set_max_radii2D(radii, visibility_filter)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, render_pkg["pixels"])
                
                prune_big_points = iteration > optim_args.opacity_reset_interval

                if iteration > optim_args.densify_from_iter:
                    if iteration % optim_args.densification_interval == 0:
                        scalars, tensors = gaussians.densify_and_prune(
                            max_grad=optim_args.densify_grad_threshold,
                            min_opacity=optim_args.min_opacity,
                            prune_big_points=prune_big_points,
                        )

                        scalar_dict.update(scalars)
                        tensor_dict.update(tensors)
                        
            # Reset opacity
            if iteration < optim_args.densify_until_iter:
                if iteration % optim_args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
                if data_args.white_background and iteration == optim_args.densify_from_iter:
                    gaussians.reset_opacity()

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