import copy
import json
import math
import os
import sys
from pathlib import Path

from dotdict import dotdict
import cv2
import hydra as hydra
from matplotlib import pyplot as plt
from torch import nn
import numpy as np
import torch

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.abspath(os.getcwd()))

from datasets.realestate10k.realestate10k_dataset import RealEstate10kDataset
from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset
from datasets.kitti_raw.kitti_raw_dataset import KittiRawDataset

from models.bts.model import BTSNet
from models.bts.model.ray_sampler import ImageRaySampler

from models.common.render import NeRFRenderer
from utils.array_operations import to, map_fn, unsqueezer
from utils.plotting import color_tensor

os.system("nvidia-smi")

device = f'cuda:0'

# DO NOT TOUCH OR YOU WILL BREAK RUNS (should be None)
gpu_id = None

if gpu_id is not None:
    print("GPU ID: " + str(gpu_id))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

r, c, = 0, 0
n_rows, n_cols = 3, 3

OUT_RES = dotdict(
    X_RANGE = (-9, 9),
    Y_RANGE = (.0, .75),
    Y_RANGE_SEGMENTATION = (0, 2),
    Y_RANGE_DEPTH_PROFILE = (0, 5),
    Z_RANGE = (21, 3),
    P_RES_ZX = (256, 256),
    P_RES_Y = 64
)


def plot(img, fig, axs, i=None):
    global r, c
    if r == 0 and c == 0:
        plt.show()
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2))
    axs[r][c].imshow(img, interpolation="none")
    if i is not None:
        axs[r][c].title.set_text(f"{i}")
    c += 1
    r += c // n_cols
    c %= n_cols
    r %= n_rows
    return fig, axs


def save_plot(img, file_name=None, grey=False, mask=None, dry_run=False):
    if mask is not None:
        if mask.shape[-1] != img.shape[-1]:
            mask = np.broadcast_to(np.expand_dims(mask, -1), img.shape)
        img = np.array(img)
        img[~mask] = 0
    if dry_run:
        plt.imshow(img)
        plt.title(file_name)
        plt.show()
    else:
        cv2.imwrite(file_name, cv2.cvtColor((img * 255).clip(max=255).astype(np.uint8), cv2.COLOR_RGB2BGR) if not grey else (img * 255).clip(max=255).astype(np.uint8))


def get_pts(x_range, y_range, z_range, x_res, y_res, z_res, cam_incl_adjust=None):
    x = torch.linspace(x_range[0], x_range[1], x_res).view(1, 1, x_res).expand(y_res, z_res, -1)
    z = torch.linspace(z_range[0], z_range[1], z_res).view(1, z_res, 1).expand(y_res, -1, x_res)
    y = torch.linspace(y_range[0], y_range[1], y_res).view(y_res, 1, 1).expand(-1, z_res, x_res)
    xyz = torch.stack((x, y, z), dim=-1)

    # The KITTI 360 cameras have a 5 degrees negative inclination. We need to account for that.
    if cam_incl_adjust is not None:
        xyz = xyz.view(-1, 3)
        xyz_h = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=-1)
        xyz_h = (cam_incl_adjust.squeeze() @ xyz_h.mT).mT
        xyz = xyz_h[:, :3].view(y_res, z_res, x_res, 3)

    return xyz


def setup_kitti360(out_folder, split="test", split_name="seg"):
    resolution = (192, 640)

    dataset = Kitti360Dataset(
        data_path="data/KITTI-360",
        pose_path="data/KITTI-360/data_poses",
        split_path=f"datasets/kitti_360/splits/{split_name}/{split}_files.txt",
        return_fisheye=False,
        return_stereo=False,
        return_depth=False,
        frame_count=1,
        target_image_size=resolution,
        fisheye_rotation=(25, -25),
        color_aug=False)

    config_path = "exp_kitti_360"

    cp_path = Path(f"out/kitti_360/pretrained")
    cp_name = cp_path.name
    cp_path = next(cp_path.glob("training*.pt"))

    out_path = Path(f"media/{out_folder}/kitti_360/{cp_name}")

    cam_incl_adjust = torch.tensor(
    [  [1.0000000,  0.0000000,  0.0000000, 0],
       [0.0000000,  0.9961947, -0.0871557, 0],
       [0.0000000,  0.0871557,  0.9961947, 0],
       [0.0000000,  000000000,  0.0000000, 1]
    ],
    dtype=torch.float32).view(1, 4, 4)

    return dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust


def setup_kittiraw(out_folder, split="test"):
    resolution = (192, 640)

    dataset = KittiRawDataset(
        data_path="data/KITTI-Raw",
        pose_path="datasets/kitti_raw/out",
        split_path=f"datasets/kitti_raw/splits/eigen_zhou/{split}_files.txt",
        frame_count=1,
        target_image_size=resolution,
        return_stereo=True,
        return_depth=False,
        color_aug=False)

    config_path = "exp_kitti_raw"

    cp_path = Path(f"out/kitti_raw/pretrained")
    cp_name = cp_path.name
    cp_path = next(cp_path.glob("training*.pt"))

    out_path = Path(f"media/{out_folder}/kitti_raw/{cp_name}")

    cam_incl_adjust = None

    return dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust


def setup_re10k(out_folder, split="test"):
    resolution = (256, 384)

    dataset = RealEstate10kDataset(
        data_path="data/RealEstate10K",
        split_path=f"datasets/realestate10k/splits/mine/{split}_files.txt" if split != "train" else None,
        frame_count=1,
        target_image_size=resolution)

    config_path = "exp_re10k"

    cp_path = Path(f"out/re10k/pretrained")
    cp_name = cp_path.name
    cp_path = next(cp_path.glob("training*.pt"))

    out_path = Path(f"media/{out_folder}/re10k/{cp_name}")

    cam_incl_adjust = None

    return dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust


def render_poses(renderer, ray_sampler, poses, projs, black_invalid=False):
    all_rays, _ = ray_sampler.sample(None, poses[:, :1], projs[:, :1])
    render_dict = renderer(all_rays, want_weights=True, want_alphas=True)

    render_dict["fine"] = dict(render_dict["coarse"])
    render_dict = ray_sampler.reconstruct(render_dict)

    depth = render_dict["coarse"]["depth"].squeeze(1)[0].cpu()
    frame = render_dict["coarse"]["rgb"][0].cpu()

    invalid = (render_dict["coarse"]["invalid"].squeeze(-1) * render_dict["coarse"]["weights"]).sum(-1).squeeze() > .8

    if black_invalid:
        depth[invalid] = depth.max()
        frame[invalid.unsqueeze(0).unsqueeze(-1), :] = 0

    return frame, depth

def render_segmentation(renderer, ray_sampler, poses, projs):
    all_rays, _ = ray_sampler.sample(None, poses[:, :1], projs[:, :1])
    render_dict = renderer(all_rays, want_weights=True, want_alphas=True, predict_segmentation=True)

    render_dict["fine"] = dict(render_dict["coarse"])
    render_dict = ray_sampler.reconstruct(render_dict, reconstruct_segmentation=True)

    segmentation = np.argmax((render_dict["coarse"]["segs"]).cpu(), axis=-1)

    return segmentation[0, 0]

def render_depth_profile(net, cam_incl_adjust):
    """Returns a top-down depth map."""
    q_pts = get_pts(OUT_RES.X_RANGE, OUT_RES.Y_RANGE_DEPTH_PROFILE, OUT_RES.Z_RANGE, OUT_RES.P_RES_ZX[1], OUT_RES.P_RES_Y,
                    OUT_RES.P_RES_ZX[0], cam_incl_adjust=cam_incl_adjust) # (num_y, num_x, num_z)

    deltas = q_pts[1:, :, :, 1] - q_pts[:-1, :, :, 1] # (n_samp - 1, num_x, num_z)
    delta_inf = 1e10 * torch.ones_like(deltas[:1, :, :])  # (1, num_x, num_z)
    deltas = torch.cat([deltas, delta_inf], 0).to(device) # (n_samp, num_x, num_z)
    q_pts = q_pts.to(device).view(1, -1, 3)

    batch_size = 50000
    if q_pts.shape[1] > batch_size:
        sigmas = []
        invalid = []
        l = q_pts.shape[1]
        for i in range(math.ceil(l / batch_size)):
            f = i * batch_size
            t = min((i + 1) * batch_size, l)
            q_pts_ = q_pts[:, f:t, :]
            _, invalid_, sigmas_, segs_ = net.forward(q_pts_, predict_segmentation=True)
            sigmas.append(sigmas_)
            invalid.append(invalid_)
        sigmas = torch.cat(sigmas, dim=1)
        invalid = torch.cat(invalid, dim=1)
    else:
        _, invalid, sigmas, segs = net.forward(q_pts, predict_segmentation=True)

    sigmas[torch.any(invalid, dim=-1)] = 1

    sigmas = sigmas.reshape(OUT_RES.P_RES_Y, *OUT_RES.P_RES_ZX)
    alphas = 1 - torch.exp(-deltas.abs() * torch.relu(sigmas))

    # (1 - alphas) with a one at index 0 for each ray (we need only need the first n_samp per ray as we never use the "full product")
    alphas_shifted = torch.cat([torch.ones_like(alphas[:1, :, :]), 1 - alphas], dim=0) # (n_samp + 1, num_x, num_z)

    T = torch.cumprod(alphas_shifted, 0) # (n_samp + 1, num_x, num_z)
    weights = alphas * T[:-1] # (n_samp, num_x, num_z)

    # we want to integrate along the ray length (not y as the camera angles means that y is not exactly top down)
    depth_profile = torch.sum(torch.linspace(0, 1, OUT_RES.P_RES_Y)[:, None, None].to(device) * weights, axis=0)

    return depth_profile

def render_segmentation_profile(net, cam_incl_adjust):
    """Note: For this to work you have to encode the image with the net first!!!"""
    q_pts = get_pts(OUT_RES.X_RANGE, OUT_RES.Y_RANGE_SEGMENTATION, OUT_RES.Z_RANGE, OUT_RES.P_RES_ZX[1], OUT_RES.P_RES_Y,
                    OUT_RES.P_RES_ZX[0], cam_incl_adjust=cam_incl_adjust) # (num_y, num_x, num_z)

    deltas = q_pts[1:, :, :, 1] - q_pts[:-1, :, :, 1] # (n_samp - 1, num_x, num_z)
    delta_inf = 1e10 * torch.ones_like(deltas[:1, :, :])  # infty (1, num_x, num_z)
    deltas = torch.cat([deltas, delta_inf], 0).to(device) # (n_samp, num_x, num_z)
    q_pts = q_pts.to(device).view(1, -1, 3)

    batch_size = 50000
    if q_pts.shape[1] > batch_size:
        sigmas = []
        invalid = []
        segs = []
        l = q_pts.shape[1]
        for i in range(math.ceil(l / batch_size)):
            f = i * batch_size
            t = min((i + 1) * batch_size, l)
            q_pts_ = q_pts[:, f:t, :]
            _, invalid_, sigmas_, segs_ = net.forward(q_pts_, predict_segmentation=True)
            sigmas.append(sigmas_)
            invalid.append(invalid_)
            segs.append(segs_)
        sigmas = torch.cat(sigmas, dim=1)
        invalid = torch.cat(invalid, dim=1)
        segs = torch.cat(segs, dim=1)
    else:
        _, invalid, sigmas, segs = net.forward(q_pts, predict_segmentation=True)

    sigmas[torch.any(invalid, dim=-1)] = 1

    sigmas = sigmas.reshape(OUT_RES.P_RES_Y, *OUT_RES.P_RES_ZX)
    alphas = 1 - torch.exp(-deltas.abs() * torch.relu(sigmas))

    # (1 - alphas) with a one at index 0 for each ray (we need only need the first n_samp per ray as we never use the "full product")
    alphas_shifted = torch.cat([torch.ones_like(alphas[:1, :, :]), 1 - alphas], dim=0) # (n_samp + 1, num_x, num_z)

    T = torch.cumprod(alphas_shifted, 0) # (n_samp + 1, num_x, num_z)
    weights = alphas * T[:-1] # (n_samp, num_x, num_z)


    segs = segs.reshape(OUT_RES.P_RES_Y, *OUT_RES.P_RES_ZX, segs.shape[-1])

    pred_profile = torch.sum(segs * weights.unsqueeze(-1), axis=0)

    return torch.argmax(pred_profile, axis=-1)

def render_profile(net, cam_incl_adjust):
    """Note: For this to work you have to encode the image with the net first!!!"""
    q_pts = get_pts(OUT_RES.X_RANGE, OUT_RES.Y_RANGE, OUT_RES.Z_RANGE, OUT_RES.P_RES_ZX[1], OUT_RES.P_RES_Y, OUT_RES.P_RES_ZX[0], cam_incl_adjust=cam_incl_adjust)
    q_pts = q_pts.to(device).view(1, -1, 3)

    batch_size = 50000
    if q_pts.shape[1] > batch_size:
        sigmas = []
        invalid = []
        l = q_pts.shape[1]
        for i in range(math.ceil(l / batch_size)):
            f = i * batch_size
            t = min((i + 1) * batch_size, l)
            q_pts_ = q_pts[:, f:t, :]
            _, invalid_, sigmas_ = net.forward(q_pts_)
            sigmas.append(sigmas_)
            invalid.append(invalid_)
        sigmas = torch.cat(sigmas, dim=1)
        invalid = torch.cat(invalid, dim=1)
    else:
        _, invalid, sigmas = net.forward(q_pts)

    sigmas[torch.any(invalid, dim=-1)] = 1
    alphas = sigmas

    alphas = alphas.reshape(OUT_RES.P_RES_Y, *OUT_RES.P_RES_ZX)

    alphas_sum = torch.cumsum(alphas, dim=0)
    profile = (alphas_sum <= 8).float().sum(dim=0) / alphas.shape[0]
    return profile


print("+++ Inference Setup Complete +++")

