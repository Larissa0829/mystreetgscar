import torch 
import os
import json
from tqdm import tqdm
from lib.models.street_gaussian_model import StreetGaussianModel 
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state
from lib.config import cfg
from lib.visualizers.base_visualizer import BaseVisualizer as Visualizer
from lib.visualizers.street_gaussian_visualizer import StreetGaussianVisualizer
import time
import sys


# 第四章 补全车辆 引入difix3d
# 添加 Difix3D/src 目录到 Python 路径
fix3d_src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Difix3D', 'src')
sys.path.append(fix3d_src_path)


def render_sets():
    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        times = []
        if not cfg.eval.skip_train:
            save_dir = os.path.join(cfg.model_path, 'train', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Training View")):
                
                torch.cuda.synchronize()
                start_time = time.time()
                result = renderer.render(camera, gaussians)
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
                visualizer.visualize(result, camera)

        if not cfg.eval.skip_test:
            save_dir = os.path.join(cfg.model_path, 'test', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras =  scene.getTestCameras()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Testing View")):
                
                torch.cuda.synchronize()
                start_time = time.time()
                
                result = renderer.render(camera, gaussians)
                                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
                visualizer.visualize(result, camera)
        
        print(times)        
        print('average rendering time: ', sum(times[1:]) / len(times[1:]))
                
def render_trajectory(custom_rotation=None, custom_translation=None, difixPipe=None, difixPrompt=None):
    with torch.no_grad():
        dataset = Dataset()        
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        
        save_dir = os.path.join(cfg.model_path, cfg.render_dir_name, "ours_{}".format(scene.loaded_iter)) #移动车辆
        visualizer = StreetGaussianVisualizer(save_dir)
        
        train_cameras = scene.getTrainCameras()
        train_cameras = list(sorted(train_cameras, key=lambda x: x.id))
        test_cameras = scene.getTestCameras()
        test_cameras = list(sorted(test_cameras, key=lambda x: x.id))
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))

        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            result = renderer.render_all(camera, gaussians,custom_rotation=custom_rotation,custom_translation=custom_translation)  #移动车辆
            visualizer.visualize(result, camera, difixPipe = difixPipe, difixPrompt = difixPrompt)
   
        # save the train and test respectively
        for idx, train_camera in enumerate(tqdm(train_cameras, desc="Rendering train_cameras Trajectory")):
            result_train_camera = renderer.render_all(train_camera, gaussians, custom_rotation=custom_rotation,custom_translation=custom_translation)  
            visualizer.visualize(result_train_camera, train_camera, mode = "train", difixPipe = difixPipe, difixPrompt = difixPrompt)

        for idx, test_camera in enumerate(tqdm(test_cameras, desc="Rendering test_cameras Trajectory")):
            result_test_camera = renderer.render_all(test_camera, gaussians, custom_rotation=custom_rotation,custom_translation=custom_translation)  
            visualizer.visualize(result_test_camera, test_camera, mode = "test", difixPipe = difixPipe, difixPrompt = difixPrompt)

        visualizer.summarize()
            
if __name__ == "__main__":
    print("Rendering " + cfg.model_path)
    safe_state(cfg.eval.quiet)

    custom_rotation = torch.tensor([cfg.render_rotate_z], dtype=torch.float32) * torch.pi / 180
    custom_translation = torch.tensor([cfg.render_move_y], dtype=torch.float32)

    difixPipe = None
    difixPrompt = None
    if cfg.optim.isDifix:
        from pipeline_difix import DifixPipeline
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 有两种，一种是不需要ref的
        difixPipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
        # 一种是需要ref的
        # difixPipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
        difixPipe.to(device)
        difixPrompt = "remove degradation"

    
    if cfg.mode == 'evaluate':
        render_sets()
    elif cfg.mode == 'trajectory':
        render_trajectory(custom_rotation, custom_translation, difixPipe, difixPrompt)
    else:
        raise NotImplementedError()
