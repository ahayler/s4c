import argparse
import random
import sys

from omegaconf import open_dict

import matplotlib.pyplot as plt
sys.path.append(".")
sys.path.extend([".", "../../../"])

import logging

from pathlib import Path
import subprocess
import yaml

import glob

import cv2
import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from hydra import compose, initialize

import matplotlib.pyplot as plt

# from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset

from models.bts.model import BTSNet, ImageRaySampler
from models.common.render import NeRFRenderer

from fusion import TSDFVolume, rigid_transform

from sscbench_dataset import SSCBenchDataset

# for every output file, there needs to be a target file with the same id; fr
#OUTPUTS_PATH = "/storage/slurm/hayler/sscbench/outputs/testset"
#OUTPUTS_PATH = "/storage/slurm/hayler/sscbench/outputs/testset10"
# OUTPUTS_PATH = "/storage/slurm/hayler/sscbench/outputs/lmscnet"
# OUTPUTS_PATH = "/storage/slurm/hayler/sscbench/outputs/sscnet"
#OUTPUTS_PATH = "/storage/slurm/hayler/sscbench/outputs/monoscene"
# TARGET_PATH = "/storage/slurm/hayler/sscbench/dataset/kitti360/KITTI-360/preprocess_new/labels/2013_05_28_drive_0009_sync"

SIZE = 51.2 # Can be: 51.2, 25.6, 12.8
SIZES = (12.8, 25.6, 51.2)

USE_ADDITIONAL_INVALIDS = True

# do we want to only eval inside the FOV
FOV_MASK_EVALUATION = True

# Setup of CUDA device and logging
os.system("nvidia-smi")

device = f'cuda:0'

gpu_id = None

if gpu_id is not None:
    print("GPU ID: " + str(gpu_id))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser("SSCBench evaluate saved outputs")
    parser.add_argument("--target_path", "-t", type=str, required=True)
    parser.add_argument("--outputs_path", "-o", type=str, required=True)

    args = parser.parse_args()

    target_path = args.target_path
    outputs_path = args.outputs_path

    with open("label_maps.yaml", "r") as f:
        label_maps = yaml.safe_load(f)

    logging.info("Loading the Lidar to Camera matrices...")

    calib = read_calib()
    T_velo_2_cam = calib["Tr"]

    logging.info("Generating the point cloud...")

    _, fov_mask = generate_point_grid(vox_origin=np.array([0, -25.6, -2]),
                              scene_size=(51.2, 51.2, 6.4),
                              voxel_size=0.2,
                              cam_E=T_velo_2_cam,
                              cam_k=get_cam_k())

    fov_mask = fov_mask.reshape(256, 256, 32)

    if not FOV_MASK_EVALUATION:
        fov_mask = np.ones([256, 256, 32])

    logging.info("Setting up folders...")

    results = {}
    for size in SIZES:
        results[size] = {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "tp_seg": np.zeros(15),
            "fp_seg": np.zeros(15),
            "tn_seg": np.zeros(15),
            "fn_seg": np.zeros(15),
        }

    # outputs = sorted(glob.glob(OUTPUTS_PATH + "/*"))
    # don't sort files for faster convergence
    # outputs = glob.glob(OUTPUTS_PATH + "/*")
    outputs = glob.glob(outputs_path + "/*")
    random.shuffle(outputs)

    # outputs = outputs[:100]

    pbar = tqdm(range(len(outputs)))

    images = {"ids": [], "images": []}

    # ids = [125, 280, 960, 1000, 1150, 1325, 2300, 3175, 3750, 4300, 5155, 5475, 5750, 6475, 6525, 6670, 6775, 7500, 7860, 8000, 8350, 9000, 9350, 10975]
    # plot_image_at_frame_id(dataset, 952)

    for i in pbar:
        output_path = outputs[i]

        frameId = output_path.split("/")[-1].split(".")[0]
        # gt_path = TARGET_PATH + f"/{frameId}_1_1.npy"
        gt_path = target_path + f"/{frameId}_1_1.npy"
        segs = np.load(output_path)

        # every gt frame_id is devisible by 5, but they have choosen to skip a few of all frame_ids that are divisible by 5, so I need to include this
        if not os.path.isfile(gt_path):
            print(f'\n Skipped frameId {frameId}, because no ground truth exists!')
            continue

        target = np.load(gt_path)

        # convert both to the right format
        segs = convert_voxels(segs, label_maps["sscbench_to_label"])
        target = convert_voxels(target, label_maps["sscbench_to_label"])

        if USE_ADDITIONAL_INVALIDS:
            invalids = identify_additional_invalids(target)
            # logging.info(np.mean(invalids))
            target[invalids == 1] = 255

        for size in SIZES:
            num_voxels = int(size // 0.2)

            # resize to right scene size
            _segs = segs[:num_voxels, (128 - num_voxels//2):(128 + num_voxels//2), :]
            _target = target[:num_voxels, (128 - num_voxels//2):(128 + num_voxels//2), :]
            _fov_mask = fov_mask[:num_voxels, (128 - num_voxels // 2):(128 + num_voxels // 2), :]

            _tp, _fp, _tn, _fn = compute_occupancy_numbers(y_pred=_segs, y_true=_target, fov_mask=_fov_mask)
            _tp_seg, _fp_seg, _tn_seg, _fn_seg = compute_occupancy_numbers_segmentation(
                y_pred=_segs, y_true=_target, fov_mask=_fov_mask, labels=label_maps["labels"])

            results[size]["tp"] += _tp
            results[size]["fp"] += _fp
            results[size]["tn"] += _tn
            results[size]["fn"] += _fn

            results[size]["tp_seg"] += _tp_seg
            results[size]["fp_seg"] += _fp_seg
            results[size]["tn_seg"] += _tn_seg
            results[size]["fn_seg"] += _fn_seg

            recall = results[size]["tp"] / (results[size]["tp"] + results[size]["fn"])
            precision = results[size]["tp"] / (results[size]["tp"] + results[size]["fp"])
            iou = results[size]["tp"] / (results[size]["tp"] + results[size]["fp"] + results[size]["fn"])

        pbar.set_postfix_str(f"IoU: {iou*100:.2f} Prec: {precision*100:.2f} Rec: {recall*100:.2f}")

    results_table = np.zeros((19, 3), dtype=np.float32)

    # Here we compute all the metrics
    for size_i, size in enumerate(SIZES):
        recall = results[size]["tp"] / (results[size]["tp"] + results[size]["fn"])
        precision = results[size]["tp"] / (results[size]["tp"] + results[size]["fp"])
        iou = results[size]["tp"] / (results[size]["tp"] + results[size]["fp"] + results[size]["fn"])

        results_table[0, size_i] = iou
        results_table[1, size_i] = precision
        results_table[2, size_i] = recall

        logging.info(f"#" * 50)
        logging.info(f"Results for size {size}. ")
        logging.info(f"#" * 50)

        logging.info("Occupancy metrics")
        logging.info(f"Recall: {recall*100:.2f}%")
        logging.info(f"Precision: {precision*100:.2f}%")
        logging.info(f"IoU: {iou*100:.2f}")

        recall_seg = results[size]["tp_seg"] / (results[size]["tp_seg"] + results[size]["fn_seg"])
        precision_seg = results[size]["tp_seg"] / (results[size]["tp_seg"] + results[size]["fp_seg"])
        iou_seg = results[size]["tp_seg"] / (results[size]["tp_seg"] + results[size]["fp_seg"] + results[size]["fn_seg"])

        weights = label_maps["weights"]
        weights_val = np.array(list(weights.values()))
        weighted_mean_iou = np.sum(weights_val * np.nan_to_num(iou_seg)) / np.sum(weights_val)

        mean_iou = np.mean(np.nan_to_num(iou_seg))

        results_table[3, size_i] = mean_iou
        results_table[4:, size_i] = iou_seg

        logging.info("Occupancy metrics segmentation")
        for i in range(15):
            logging.info(f"{label_maps['labels'][i+1]}; IoU: {iou_seg[i]*100:.2f}; Precision: {precision_seg[i]*100:.2f}%; Recall: {recall_seg[i]*100:.2f}%")

    logging.info(f"Results table for copying.")

    results_table_str = ""
    for i in range(19):
        results_table_str += f"{results_table[i, 0]*100:.2f}\t{results_table[i, 1]*100:.2f}\t{results_table[i, 2]*100:.2f}\n"
    print(results_table_str)

    # paper = [29.41, 2.73, 1.97, 6.08, 3.71, 2.86, 66.10, 18.44, 38.00, 4.49, 41.12, 8.99, 45.68, 24.70, 8.84, 9.15, 10.31, 4.4]

    logging.info(f"Mean IoU: {mean_iou*100:.2f}")
    logging.info(f"Weighted Mean IoU: {weighted_mean_iou*100:.2f}")


def identify_additional_invalids(target):
    # Note: The Numpy implementation is a bit faster (about 0.1 seconds per iteration)

    _t = np.concatenate([np.zeros([256, 256, 1]), target], axis=2)
    invalids = np.cumsum(np.logical_and(_t != 255, _t != 0), axis=2)[:, :, :32] == 0
    # _t = torch.cat([torch.zeros([256, 256, 1], device=device, dtype=torch.int32), torch.tensor(target, dtype=torch.int32).to(device)], dim=2)
    # invalids = torch.cumsum((_t != 255) & (_t != 0), axis=2)[:,:, :32] == 0
    # height cut-off (z > 6 ==> no invalid)
    invalids[: , :, 7:] = 0
    # only empty voxels matter
    invalids[target != 0] = 0

    # return invalids.cpu().numpy()
    return invalids


def generate_point_grid(cam_E, vox_origin, voxel_size, scene_size, cam_k, img_W=1408, img_H=376):
        """
        compute the 2D projection of voxels centroids

        Taken from: https://github.com/ai4ce/SSCBench

        Parameters:
        ----------
        cam_E: 4x4
           =camera pose in case of NYUv2 dataset
           =Transformation from camera to lidar coordinate in case of SemKITTI
        cam_k: 3x3
            camera intrinsics
        vox_origin: (3,)
            world(NYU)/lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
        img_W: int
            image width
        img_H: int
            image height
        scene_size: (3,)
            scene size in meter: (51.2, 51.2, 6.4) for SemKITTI and (4.8, 4.8, 2.88) for NYUv2

        Returns
        -------
        projected_pix: (N, 2)
            Projected 2D positions of voxels
        fov_mask: (N,)
            Voxels mask indice voxels inside image's FOV
        pix_z: (N,)
            Voxels'distance to the sensor in meter
        """
        # Compute the x, y, z bounding of the scene in meter
        vol_bnds = np.zeros((3, 2))
        vol_bnds[:, 0] = vox_origin
        vol_bnds[:, 1] = vox_origin + np.array(scene_size)

        # Compute the voxels centroids in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).copy(order='C').astype(int)
        xv, yv, zv = np.meshgrid(
            range(vol_dim[0]),
            range(vol_dim[1]),
            range(vol_dim[2]),
            indexing='ij'
        )
        vox_coords = np.concatenate([
            xv.reshape(1, -1),
            yv.reshape(1, -1),
            zv.reshape(1, -1)
        ], axis=0).astype(int).T

        # Project voxels'centroid from lidar coordinates to camera coordinates
        cam_pts = TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)
        cam_pts = rigid_transform(cam_pts, cam_E)

        # Project camera coordinates to pixel positions
        projected_pix = TSDFVolume.cam2pix(cam_pts, cam_k)
        pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

        # Eliminate pixels outside view frustum
        pix_z = cam_pts[:, 2]
        fov_mask = np.logical_and(pix_x >= 0,
                                  np.logical_and(pix_x < img_W,
                                                 np.logical_and(pix_y >= 0,
                                                                np.logical_and(pix_y < img_H,
                                                                               pix_z > 0))))

        return cam_pts, fov_mask


def convert_voxels(arr, map_dict):
    f = np.vectorize(map_dict.__getitem__)
    return f(arr)


def compute_occupancy_numbers_segmentation(y_pred, y_true, fov_mask, labels):
    label_ids = list(labels.keys())[1:]
    mask = y_true != 255
    mask = np.logical_and(mask, fov_mask)
    mask = mask.flatten()

    y_pred = y_pred.flatten()[mask]
    y_true = y_true.flatten()[mask]

    tp = np.zeros(len(label_ids))
    fp = np.zeros(len(label_ids))
    fn = np.zeros(len(label_ids))
    tn = np.zeros(len(label_ids))

    for label_id in label_ids:
        tp[label_id - 1] = np.sum(np.logical_and(y_true == label_id, y_pred == label_id))
        fp[label_id - 1] = np.sum(np.logical_and(y_true != label_id, y_pred == label_id))
        fn[label_id - 1] = np.sum(np.logical_and(y_true == label_id, y_pred != label_id))
        tn[label_id - 1] = np.sum(np.logical_and(y_true != label_id, y_pred != label_id))

    return tp, fp, tn, fn


def compute_occupancy_numbers(y_pred, y_true, fov_mask):
    mask = y_true != 255
    mask = np.logical_and(mask, fov_mask)
    mask = mask.flatten()

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    occ_true = y_true[mask] > 0
    occ_pred = y_pred[mask] > 0

    tp = np.sum(np.logical_and(occ_true == 1, occ_pred == 1))
    fp = np.sum(np.logical_and(occ_true == 0, occ_pred == 1))
    fn = np.sum(np.logical_and(occ_true == 1, occ_pred == 0))
    tn = np.sum(np.logical_and(occ_true == 0, occ_pred == 0))

    return tp, fp, tn, fn


def read_calib():
    """
    :param calib_path: Path to a calibration text file.
    :return: dict with calibration matrices.

    taken/modified from: https://github.com/ai4ce/SSCBench
    """
    P = np.array(
        [
            552.554261,
            0.000000,
            682.049453,
            0.000000,
            0.000000,
            552.554261,
            238.769549,
            0.000000,
            0.000000,
            0.000000,
            1.000000,
            0.000000,
        ]
    ).reshape(3, 4)

    cam2velo = np.array(
        [
            0.04307104361,
            -0.08829286498,
            0.995162929,
            0.8043914418,
            -0.999004371,
            0.007784614041,
            0.04392796942,
            0.2993489574,
            -0.01162548558,
            -0.9960641394,
            -0.08786966659,
            -0.1770225824,
        ]
    ).reshape(3, 4)
    C2V = np.concatenate(
        [cam2velo, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0
    )
    # print("C2V: ", C2V)
    V2C = np.linalg.inv(C2V)
    # print("V2C: ", V2C)
    V2C = V2C[:3, :]
    # print("V2C: ", V2C)

    # reshape matrices
    calib_out = {}
    # 3x4 projection matrix for left camera
    calib_out["P2"] = P
    calib_out["Tr"] = np.identity(4)  # 4x4 matrix
    calib_out["Tr"][:3, :4] = V2C
    return calib_out


def get_cam_k():
    """
    taken/modified from: https://github.com/ai4ce/SSCBench
    """
    cam_k = np.array(
        [
            552.554261,
            0.000000,
            682.049453,
            0.000000,
            0.000000,
            552.554261,
            238.769549,
            0.000000,
            0.000000,
            0.000000,
            1.000000,
            0.000000,
        ]
    ).reshape(3, 4)
    return cam_k[:3, :3]


if __name__ == "__main__":
    main()
