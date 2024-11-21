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

import re
import os
import torch
import torch.nn as nn
from random import randint
from utils.loss_utils import l1_loss, l1_loss_sum, ssim
from gaussian_renderer import network_gui#, render
from gaussian_renderer.render_comp import render,render_obj,render_depth 
import sys
from scene import Scene_comp, Scene_comp_multi, GaussianModel
from scene.gaussian_model_multidyn import GaussianModelDyn
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import logging
from datetime import datetime
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
current_date = datetime.now().strftime('%Y%m%d%H%M%S')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(logger_dir):
    os.mkdir(logger_dir) 
logger_path = os.path.join(logger_dir, "train_comp_0802_symm.log")
if os.path.exists(logger_path):  # 确保文件存在
    os.remove(logger_path)  # 删除文件
handler = logging.FileHandler(logger_path)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians1 = GaussianModel(dataset.sh_degree)
    gaussians2 = GaussianModelDyn(dataset.sh_degree)
    scene = Scene_comp_multi(dataset, gaussians1,gaussians2)
    gaussians1.training_setup(opt)
    gaussians2.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    loss_dyn = nn.CrossEntropyLoss()
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians1.update_learning_rate(iteration)
        gaussians2.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians1.oneupSHdegree()
            gaussians2.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gaussians2.o2w_ref(viewpoint_cam.obj_rt,viewpoint_cam.obj_quat)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians1, gaussians2, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        image_dyn = render_obj(viewpoint_cam, gaussians2, pipe, background)
        #image_dyn = render_obj_update(viewpoint_cam, gaussians1, gaussians2, pipe, background)
        image_dyn[:,1] = 1 - image_dyn[:,0]

        # depth loss 
        image_depth = render_depth(viewpoint_cam, gaussians1, pipe, background)
        loss_depth = l1_loss_sum(image_depth * viewpoint_cam.depth[...,1],viewpoint_cam.depth[...,0])/viewpoint_cam.depth[...,1].sum()
        #print(viewpoint_cam.depth[...,1].max())
        #quit()

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # add, by yyf
        mask = viewpoint_cam.gt_alpha_mask
        # gt_image = gt_image * mask 
        # image = image * mask
        mask = mask.reshape(-1)
        mask_new = mask[:,None].repeat(1,2)
        mask_new[:,1] = 1 - mask 
        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 0.2 * loss_dyn(image_dyn,mask_new) + 0.05 * loss_depth
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}"})
                progress_bar.update(10)
                logger.info('Iteration:{}'.format(iteration)+' Loss:'+f"{ema_loss_for_log:.{7}f}")
                #print(gaussians2.label.shape,gaussians2.label.sum().item(),gaussians2.label.max().item(),gaussians2.label.min().item())
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                visibility_filter_1,visibility_filter_2 = visibility_filter[:gaussians1._xyz.shape[0]],visibility_filter[gaussians1._xyz.shape[0]:gaussians1._xyz.shape[0]+gaussians2._xyz.shape[0]]
                radii_1,radii_2 = radii[:gaussians1._xyz.shape[0]],radii[gaussians1._xyz.shape[0]:gaussians1._xyz.shape[0]+gaussians2._xyz.shape[0]]

                gaussians1.max_radii2D[visibility_filter_1] = torch.max(gaussians1.max_radii2D[visibility_filter_1], radii_1[visibility_filter_1])
                gaussians1.add_densification_stats_comp(viewspace_point_tensor, visibility_filter_1,[i for i in range(gaussians1._xyz.shape[0])])
                gaussians2.max_radii2D[visibility_filter_2] = torch.max(gaussians2.max_radii2D[visibility_filter_2], radii_2[visibility_filter_2])
                gaussians2.add_densification_stats_comp(viewspace_point_tensor, visibility_filter_2,[i for i in range(gaussians1._xyz.shape[0],gaussians1._xyz.shape[0]+gaussians2._xyz.shape[0])])
                
                #print(scene.cameras_extent)
                #quit()
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    #size_threshold1 = 20 if iteration > opt.opacity_reset_interval else None
                    size_threshold2 = 20 if iteration > opt.opacity_reset_interval else None
                    size_threshold1 = None
                    #size_threshold2 = None
                    #print(gaussians1.max_radii2D.max())
                    #print(gaussians2.max_radii2D.max())
                    gaussians1.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold1)
                    gaussians2.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold2)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians1.reset_opacity()
                    gaussians2.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians1.optimizer.step()
                gaussians1.optimizer.zero_grad(set_to_none = True)
                gaussians2.optimizer.step()
                gaussians2.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        pattern = re.compile("/([^/]+)$")
        result = pattern.findall(args.source_path)
        args.model_path = os.path.join("./output/", result[0]+'_update_0802_symm')#unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene_comp_multi, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    scene.gaussians2.o2w_ref(viewpoint.obj_rt,viewpoint.obj_quat)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians1, scene.gaussians2, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                logger.info("[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
