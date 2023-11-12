import os
import os
import sys
from pathlib import Path

import cv2
import hydra as hydra
import torch.nn.functional as F
from matplotlib import pyplot as plt
from utils.plotting import color_segmentation_tensor
from omegaconf import open_dict
from torch import nn
from tqdm import tqdm


sys.path.append(os.path.abspath(os.getcwd()))

from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset
from torch.utils.data import DataLoader

from models.bts.model import BTSNet

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from datasets.kitti_raw.kitti_raw_dataset import KittiRawDataset

from models.common.render import NeRFRenderer

os.system("nvidia-smi")

config_path = "exp_kitti_360"
# cp_path = Path(f"../../out/kitti_360/pretrained")
# cp_path = Path(f"out/kitti_360/k360_spatial_pix-mlp_d+c_mod_backend-None-1_20230319-173622")
# cp_path = Path("../../out/kitti_360/kitti_360_backend-None-1_20230626-221220") # proper model nsegs=8 pamoptic deeplab
cp_path = Path("../../out/kitti_360/kitti_360_backend-None-1_20230712-173222")
cp_path = next(cp_path.glob("training*.pt"))

out_path = Path("../../media/voxel/presentation/kitti_360_val")
out_path.mkdir(exist_ok=True, parents=True)

show = False
save = True
s_img = True
s_depth = True
dry_run = False

resolution = (192, 640)
# resolution = (128, 356)

indices_3d_short = [0, 1, 2, 3, 4]
indices_video = list(range(1000, 1290, 1))
indices_presentation = [8636, 8638, 8640, 9402, 9999, 10165, 11153]
# indices_presentation = [9399]
# indices_presentation = [9399]
indices_presentation = [159] # overfit picture

X_RANGE = (-9, 9)
Y_RANGE = (-0.6, 1.8)
# Y_RANGE = (-6, 2)
Z_RANGE = (21, 3)

p_res = (128, 128)
p_res_y = 16

# scale_factor = 2
#
# p_res = (int(scale_factor * 128), int(scale_factor * 128))
# p_res_y = int(scale_factor * 16)

# gpu_id = 0
gpu_id = 1

device = f'cuda:0'
if gpu_id is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

r, c, = 0, 0
n_rows, n_cols = 3, 3


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


def save_plot(img, file_name=None, grey=False, mask=None):
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


# faces = [[0, 2, 1], [3, 1, 2],
#          [0, 1, 4], [5, 4, 1],
#          [0, 4, 2], [6, 2, 4],
#          [1, 3, 5], [7, 5, 3],
#          [2, 6, 3], [7, 3, 6],
#          [4, 5, 6], [7, 6, 5]]

faces = [[0, 1, 2, 3], [0, 3, 7, 4], [2, 6, 7, 3], [1, 2, 6, 5], [0, 1, 5, 4], [4, 5, 6, 7]]
faces_t = torch.tensor(faces, device=device)

y_steps = (1 - (torch.linspace(0, 1 - 1/p_res_y, p_res_y) + 1 / (2 * p_res_y))).tolist()
cmap = plt.cm.get_cmap("magma")
y_to_color = (torch.tensor(list(map(cmap, y_steps)), device=device)[:, :3] * 255).to(torch.uint8)
N_CLASSES = 21
class_to_color = torch.tensor(plt.cm.plasma(np.linspace(0, 1, N_CLASSES))).to(device)
class_to_color = (class_to_color[:, :3] * 255).to(torch.uint8) # RGBA --> RGB


def build_voxel(i, j, k, x_res, y_res, z_res, xyz, offset):
    ids = [[i+1, j+1, k], [i+1, j, k],
           [i, j, k], [i, j+1, k],
           [i+1, j+1, k+1], [i+1, j, k+1],
           [i, j, k+1], [i, j+1, k+1]]

    faces_off = [[v+offset for v in f] for f in faces]

    ids_flat = list(map(lambda ijk: ijk[0]*y_res*z_res + ijk[1]*z_res + ijk[2], ids))

    verts = xyz[:, ids_flat].cpu().numpy().T

    colors = np.tile(np.array(plt.cm.get_cmap("magma")(1 - (verts[..., 1].mean().item() - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]))[:3]).reshape((1, 3)), ((len(faces_off), 1)))
    colors = (colors * 255).astype(np.uint8)

    return verts, faces_off, colors


ids_offset = torch.tensor(
        [[1, 1, 0], [1, 0, 0],
        [0, 0, 0], [0, 1, 0],
        [1, 1, 1], [1, 0, 1],
        [0, 0, 1], [0, 1, 1]],
    dtype=torch.int32,
    device=device
) # (8, 3) # this is the offset for our cube (voxel = 3d pixel)

def build_voxels(ijks, x_res, y_res, z_res, xyz):
    # ijks (N, 3)

    ids = ijks.view(-1, 1, 3) + ids_offset.view(1, -1, 3)

    ids_flat = ids[..., 0] * y_res * z_res + ids[..., 1] * z_res + ids[..., 2]

    verts = xyz[:, ids_flat.reshape(-1)]

    faces_off = torch.arange(0, ijks.shape[0] * 8, 8, device=device)
    faces_off = faces_off.view(-1, 1, 1) + faces_t.view(-1, 6, 4)

    colors = y_to_color[ijks[:, 1], :].view(-1, 1, 3).expand(-1, 8, -1)

    return verts.cpu().numpy().T, faces_off.reshape(-1, 4).cpu().numpy(), colors.reshape(-1, 3).cpu().numpy()

def build_voxels_segmentation(ijks, x_res, y_res, z_res, xyz, classes):
    # ijks (N, 3)

    ids = ijks.view(-1, 1, 3) + ids_offset.view(1, -1, 3)

    ids_flat = ids[..., 0] * y_res * z_res + ids[..., 1] * z_res + ids[..., 2]

    verts = xyz[:, ids_flat.reshape(-1)]

    faces_off = torch.arange(0, ijks.shape[0] * 8, 8, device=device)
    faces_off = faces_off.view(-1, 1, 1) + faces_t.view(-1, 6, 4)

    classes = classes[0, 0]

    index_classes = classes[ijks[:, 0], ijks[:, 1], ijks[:, 2]]

    colors = class_to_color[index_classes].view(-1, 1, 3).expand(-1, 8, -1)
    # colors = y_to_color[ijks[:, 1], :].view(-1, 1, 3).expand(-1, 8, -1)

    return verts.cpu().numpy().T, faces_off.reshape(-1, 4).cpu().numpy(), colors.reshape(-1, 3).cpu().numpy()


def get_pts(x_range, y_range, z_range, x_res, y_res, z_res):
    x = torch.linspace(x_range[0], x_range[1], x_res).view(1, 1, x_res).expand(y_res, z_res, -1)
    z = torch.linspace(z_range[0], z_range[1], z_res).view(1, z_res, 1).expand(y_res, -1, x_res)
    y = torch.linspace(y_range[0], y_range[1], y_res).view(y_res, 1, 1).expand(-1, z_res, x_res)
    xyz = torch.stack((x, y, z), dim=-1).permute(2, 0, 1, 3)                                            # (x, y, z)

    # The KITTI 360 cameras have a 5 degrees negative inclination. We need to account for that tan(5°) = 0.0874886635
    xyz[:, :, :, 1] -= xyz[:, :, :, 2] * 0.0874886635

    return xyz


@hydra.main(version_base=None, config_path="../../configs", config_name=config_path)
def main(config):
    print('Loading dataset')

    resolution = (192, 640)

    dataset = Kitti360Dataset(
        data_path=config["data"]["data_path"],
        pose_path=config["data"]["pose_path"],
        data_segmentation_path=config["data"]["data_segmentation_path"],
        segmentation_mode='panoptic_deeplab',
        # split_path=f"datasets/kitti_360/splits/val_seq/2013_05_28_drive_0000_sync_files.txt",
        # split_path=f"../../datasets/kitti_360/splits/seg/val_files.txt",
        split_path=os.path.join("../../", config["data"]["split_path"], "val_files.txt"),
        return_fisheye=False,
        return_stereo=False,
        return_depth=False,
        frame_count=1,
        target_image_size=resolution,
        fisheye_rotation=(25, -25),
        color_aug=False)

    print('Loading checkpoint')
    cp = torch.load(cp_path, map_location=device)

    with open_dict(config):
        config["renderer"]["hard_alpha_cap"] = True
        config["model_conf"]["code_mode"] = "z"
        # config["model_conf"]["z_near"] = 8
        config["model_conf"]["mlp_coarse"]["n_blocks"] = 0
        config["model_conf"]["mlp_coarse"]["d_hidden"] = 64
        config["model_conf"]["encoder"]["d_out"] = 64
        config["model_conf"]["encoder"]["type"] = "monodepth2"
        config["model_conf"]["grid_learn_empty"] = False
        config["model_conf"]["sample_color"] = True

        # stuff for segmentation
        config["model_conf"]["segmentation_mode"] = 'panoptic_deeplab'


    net = BTSNet(config["model_conf"])
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()
    renderer.renderer.n_coarse = 256
    renderer.renderer.lindisp = False

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.renderer = renderer

    _wrapper = _Wrapper()

    _wrapper.load_state_dict(cp["model"], strict=False)
    renderer.to(device)
    renderer.eval()

    with torch.no_grad():

        for idx in tqdm(indices_presentation):
            dataset._skip = idx
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            data_batch = next(iter(loader))

            images = torch.stack(data_batch["imgs"], dim=1).to(device)
            poses = torch.stack(data_batch["poses"], dim=1).to(device)
            projs = torch.stack(data_batch["projs"], dim=1).to(device)

            poses = torch.inverse(poses[:, :1]) @ poses

            # plot the first input
            plt.imshow(((images[0, 0] + 1) / 2).cpu().permute(1, 2, 0))
            plt.show()

            plt.imshow(color_segmentation_tensor(data_batch['segs_gt'][0].squeeze(0)))
            plt.show()

            n, nv, c, h, w = images.shape

            net.compute_grid_transforms(projs, poses)
            net.encode(images, projs, poses, ids_encoder=[0], ids_render=[0])

            net.set_scale(0)

            q_pts = get_pts(X_RANGE, Y_RANGE, Z_RANGE, p_res[1], p_res_y, p_res[0])
            q_pts = q_pts.to(device).reshape(1, -1, 3)
            # _, invalid, sigmas = net.forward(q_pts)

            _, invalid, sigmas, segs = net.forward(q_pts, predict_segmentation=True)

            sigmas[torch.any(invalid, dim=-1)] = 0
            alphas = sigmas

            alphas = alphas.reshape(1, 1, p_res[1], p_res_y, p_res[0])              # (x, y, z)
            segs = segs.reshape(1, 1, p_res[1], p_res_y, p_res[0], -1)

            delta = np.mean([
                (X_RANGE[1] - X_RANGE[0]) / p_res[0],
                (Y_RANGE[1]- Y_RANGE[0]) / p_res_y,
                (Z_RANGE[0] - Z_RANGE[1]) / p_res[1]
            ])

            real_alphas = 1 - torch.exp(-delta*alphas)

            # weight the segs by density
            segs = segs * real_alphas[..., None]

            alphas_mean = F.avg_pool3d(alphas, kernel_size=2, stride=1, padding=0)
            is_occupied = alphas_mean.squeeze() > .5

            segs_pool_list = [F.avg_pool3d(segs[..., i], kernel_size=2, stride=1, padding=0) for i in
                              range(segs.shape[-1])]
            segs_pool = torch.stack(segs_pool_list, dim=-1)
            classes = torch.argmax(segs_pool, dim=-1)

            # verts, faces, colors_old = build_voxels(is_occupied.nonzero(), p_res[1], p_res_y, p_res[0], q_pts.squeeze(0).T)

            verts, faces, colors = build_voxels_segmentation(is_occupied.nonzero(), p_res[1], p_res_y, p_res[0],
                                                             q_pts.squeeze(0).T, classes)

            verts = list(map(tuple, verts))
            colors = list(map(tuple, colors))
            verts_colors = [v + c for v, c in zip(verts, colors)] # this appends the color (not elementwise addition)
            verts_data = np.array(verts_colors, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

            face_data = np.array(faces, dtype='i4')
            ply_faces = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (4,))])
            ply_faces['vertex_indices'] = face_data

            verts_el = PlyElement.describe(verts_data, "vertex")
            faces_el = PlyElement.describe(ply_faces, "face")
            # PlyData([verts_el, faces_el]).write(str(out_path / f"{idx:010d}.ply"))
            PlyData([verts_el, faces_el]).write(str(out_path / f"{idx:010d}_segmentation.ply"))

    pass


if __name__ == '__main__':
    main()
