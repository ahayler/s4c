"""
Code partly taken from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/data/labels_downscale.py
"""
import numpy as np
from tqdm import tqdm
import numpy.matlib
import os
import glob
import hydra
from omegaconf import DictConfig
import io_data as SemanticKittiIO
from hydra.utils import get_original_cwd
# from monoscene.data.NYU.preprocess import _downsample_label
from evaluate_model_sscbench import identify_additional_invalids

# OUTPUT_ROOT = "/usr/stud/hayler/dev/BehindTheScenes/scripts/benchmarks/sscbench/test"
OUTPUT_ROOT = "/storage/slurm/hayler/sscbench/dataset/kitti360/preprocessed_data/1_1_with_invalids"
KITTI_360_DATA_ROOT = "/storage/slurm/hayler/sscbench/dataset/kitti360/KITTI-360"
UPDATE_INVALIDS = True


def majority_pooling(grid, k_size=2):
    result = np.zeros(
        (grid.shape[0] // k_size, grid.shape[1] // k_size, grid.shape[2] // k_size)
    )
    for xx in range(0, int(np.floor(grid.shape[0] / k_size))):
        for yy in range(0, int(np.floor(grid.shape[1] / k_size))):
            for zz in range(0, int(np.floor(grid.shape[2] / k_size))):

                sub_m = grid[
                        (xx * k_size): (xx * k_size) + k_size,
                        (yy * k_size): (yy * k_size) + k_size,
                        (zz * k_size): (zz * k_size) + k_size,
                        ]
                unique, counts = np.unique(sub_m, return_counts=True)
                if True in ((unique != 0) & (unique != 255)):
                    # Remove counts with 0 and 255
                    counts = counts[((unique != 0) & (unique != 255))]
                    unique = unique[((unique != 0) & (unique != 255))]
                else:
                    if True in (unique == 0):
                        counts = counts[(unique != 255)]
                        unique = unique[(unique != 255)]
                value = unique[np.argmax(counts)]
                result[xx, yy, zz] = value
    return result


def main():
    scene_size = (256, 256, 32)
    sequences = ["00","01", "02", "03", "04", "05", "06", "07", "09", "10"]
    # sequences = ["00"]
    remap_lut = SemanticKittiIO.get_remap_lut("kitti_360_preprocess.yaml")

    for sequence in sequences:
        sequence_path = os.path.join(
            KITTI_360_DATA_ROOT, "data_2d_raw", "2013_05_28_drive_00" + sequence + "_sync"
        )
        # print("sequence_path: ", sequence_path)
        label_paths = sorted(
            glob.glob(os.path.join(sequence_path, "voxels", "*.label"))
        )
        invalid_paths = sorted(
            glob.glob(os.path.join(sequence_path, "voxels", "*.invalid"))
        )
        out_dir = os.path.join(OUTPUT_ROOT, "labels", "2013_05_28_drive_00" + sequence + "_sync")
        os.makedirs(out_dir, exist_ok=True)

        for i in tqdm(range(len(label_paths))):
            frame_id, extension = os.path.splitext(os.path.basename(label_paths[i]))

            LABEL = SemanticKittiIO._read_label_KITTI360(label_paths[i])
            # print("LABEL.shape: ", LABEL.shape)
            INVALID = SemanticKittiIO._read_invalid_KITTI360(invalid_paths[i])
            # print("INVALID.shape: ", INVALID.shape)
            nonzero_idx = np.nonzero(LABEL)
            INVALID[nonzero_idx] = 0
            LABEL = remap_lut[LABEL.astype(np.uint16)].astype(
                np.float32
            )  # Remap 20 classes semanticKITTI SSC

            LABEL[np.isclose(INVALID, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...

            LABEL = LABEL.reshape([256, 256, 32])

            if UPDATE_INVALIDS:
                invalids = identify_additional_invalids(LABEL)
                LABEL[invalids == 1] = 255

            filename = frame_id + "_" + "1_1" + ".npy"
            label_filename = os.path.join(out_dir, filename)

            np.save(label_filename, LABEL)
            #print("wrote to", label_filename)


if __name__ == "__main__":
    main()

