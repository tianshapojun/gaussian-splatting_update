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

import torch
from scene import Scene_comp, Scene_comp_multi
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer.render_comp import render,render_double,render_obj,render_multi
#from gaussian_renderer import render
import torchvision
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrix2
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.gaussian_model_multidyn import GaussianModelDyn

        
class cam_new(nn.Module):
    def __init__(self, c1,c2, pp
                 ):
        super(cam_new, self).__init__()

        self.R = (1-pp)*c1.R + pp*c2.R
        self.T = (1-pp)*c1.T + pp*c2.T
        self.FoVx = (1-pp)*c1.FoVx + pp*c2.FoVx
        self.FoVy = (1-pp)*c1.FoVy + pp*c2.FoVy

        self.data_device = torch.device(c1.data_device)
        
        self.image_width = c1.image_width
        self.image_height = c1.image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = c1.trans
        self.scale = c1.scale
    
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix2(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=c1.cx, cy=c1.cy, h=c1.original_image.shape[1], w=c1.original_image.shape[2]).transpose(0,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]     
     
        
def render_set_interp(model_path, name, iteration, views, gaussians1, gaussians2,pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))

    makedirs(render_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # if idx == len(views)-1:
        #     continue 
        # if idx %9 == 8: 
        #     continue 
        # for i in range(2):
        #     view_new = cam_new(views[idx],views[idx+1],0.25*i) 
        #     rendering = render(view_new, gaussians, pipeline, background)["render"]
        #     torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(rr) + ".png"))
        #     rr +=1

        rt_matrix = view.obj_rt.copy()
        if idx < 15:
            rt_matrix[2:] = views[36+idx].obj_rt[2:]
        else:
            rt_matrix[2:] = views[-1].obj_rt[2:]
            rt_matrix[1] = views[15].obj_rt[1].copy()
        # rt_matrix = view.obj_rt.copy()
        # if idx < 15:
        #     rt_matrix[2:] = views[36+idx].obj_rt[2:]
        # elif idx <33:
        #     rt_matrix[2:] = views[-1].obj_rt[2:].copy()
        #     rt_matrix[1] = views[15].obj_rt[1].copy()
        #     rt_matrix[2][0][3] += (idx-15)*0.01
        # else: 
        #     rt_matrix[1] = views[15].obj_rt[1].copy()
        #     rt_matrix[2:] = views[-1].obj_rt[2:].copy()
        #     rt_matrix[2][0][3] -= (idx-32)*0.01
        #print(views[-1].obj_rt[2][0][3])
        #gaussians2.o2w(rt_matrix)
        gaussians2.o2w_inf(rt_matrix,idx)
        #gaussians2.o2w(view.obj_rt)
        #gaussians2.o2w_inf(view.obj_rt,idx)
        #gaussians2.vehicle_clone(view.obj_rt,idx)
        #rendering = render_double(view, gaussians1, gaussians2, pipeline, background)["render"]
        rendering = render(view, gaussians1, gaussians2, pipeline, background)["render"]
        #rendering = render(view, gaussians2, pipeline, background)["render"]
        #rendering = render_obj(view, gaussians2, pipeline, background)
        #rendering = render_multi(view, gaussians1, gaussians2, 1, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians1 = GaussianModel(dataset.sh_degree)
        gaussians2 = GaussianModelDyn(dataset.sh_degree)
        scene = Scene_comp_multi(dataset, gaussians1, gaussians2,load_iteration=iteration, shuffle=False)
        #print(scene.gaussians.get_xyz.shape)
        # kk = scene.gaussians2.get_xyz 
        # print(kk.shape)
        # print(kk[:,0].max(),kk[:,0].min())
        # print(kk[:,1].max(),kk[:,1].min())
        # print(kk[:,2].max(),kk[:,2].min())
        # # kk = kk.detach().cpu().numpy()
        # # import numpy as np
        # # kk = kk.astype(np.float32)
        # # kk.tofile("gs_pcd.bin")
        #quit()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # print('\n')
        # print(scene.getTrainCameras()[1].T)
        # quit()

        render_set_interp(dataset.model_path, "eval_r", scene.loaded_iter, scene.getTrainCameras(), gaussians1, gaussians2,pipeline, background)

        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args))
