import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

print(os.getcwd())
sys.path.append('..')

import argparse
import cv2
import logging
import numpy as np
from PIL import Image, ImageOps
import torch
from tqdm import tqdm

from segmentation.config import config, update_config
from segmentation.model import build_segmentation_model_from_cfg
from segmentation.model.post_processing import get_semantic_segmentation
import segmentation.data.transforms.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

DRY_RUN = False
MODEL_RES = (513, 1697)
OUT_RES = (192, 640)


def read_image(file_name, format=None):
    image = Image.open(file_name)

    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format == "BGR":
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    if format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)
    return image


def parse_args():
    global DRY_RUN, MODEL_RES, OUT_RES

    parser = argparse.ArgumentParser(description='Test segmentation network with single process')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='configs/panoptic_deeplab_R101_os32_cityscapes.yaml',
                        type=str)
    parser.add_argument('--input-dir', '-i',
                        default='/storage/group/dataset_mirrors/01_incoming/kitti_360/KITTI-360/data_2d_raw',
                        type=str)
    parser.add_argument('--output-dir', '-o',
                        help='output directory',
                        required=True,
                        type=str)
    parser.add_argument('--model', '-m',
                        default='/storage/slurm/hayler/bts/panoptic_deeplab/panoptic_deeplab_R101_os32_cityscapes.pth',
                        type=str)
    parser.add_argument('--dry-run', '-d',
                        action='store_true')
    parser.add_argument('--model_res', default=MODEL_RES)
    parser.add_argument('--out_res', default=OUT_RES)
    parser.add_argument('--checkpoint', required=True)


    args = parser.parse_args()
    args.opts = ['TEST.MODEL_FILE', args.model]


    update_config(config, args)


    DRY_RUN = args.dry_run
    MODEL_RES = args.model_res
    OUT_RES = args.out_res

    if DRY_RUN:
        logging.warning("#### Dry run mode. ####")

    return args


def setup_model(args):
    model = build_segmentation_model_from_cfg(config)

    gpus = list(config.TEST.GPUS)
    if len(gpus) > 1:
        raise ValueError('Test only supports single core.')
    device = torch.device('cuda:{}'.format(gpus[0]))

    model = model.to(device)

    model_state_file = args.checkpoint

    # # load model
    # if config.TEST.MODEL_FILE:
    #     model_state_file = config.TEST.MODEL_FILE
    # else:
    #     model_state_file = os.path.join(config.OUTPUT_DIR, 'final_state.pth')

    if os.path.isfile(model_state_file):
        print("Loading test model file:", model_state_file)
        model_weights = torch.load(model_state_file)
        if 'state_dict' in model_weights.keys():
            model_weights = model_weights['state_dict']
        model.load_state_dict(model_weights, strict=True)
    else:
        print("Loading failed:", model_state_file)
        raise ValueError('Cannot find test model.')

    model.eval()

    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                config.DATASET.MEAN,
                config.DATASET.STD
            )
        ]
    )

    return model, device, args, transforms


def predict_image(path: str, model, transforms, device):
    raw_image = read_image(path, 'RGB')

    # input_image = cv2.resize(raw_image, (641, 193))
    input_image = cv2.resize(raw_image, (MODEL_RES[1], MODEL_RES[0]))
    image, _ = transforms(input_image, None)
    image = image.unsqueeze(0).to(device)
    out_dict = model(image)
    semantic_pred = get_semantic_segmentation(out_dict['semantic'])

    # fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    # axs[0].imshow(input_image, cmap="jet", interpolation="none")
    # axs[1].imshow(semantic_pred.cpu().squeeze(), cmap="hsv", interpolation="none", vmin=0, vmax=33)
    # axs[2].imshow(input_image, cmap="jet", interpolation="none")
    # axs[2].imshow(semantic_pred.cpu().squeeze(), cmap="hsv", interpolation="none", vmin=0, vmax=33, alpha=0.5)
    # plt.show()

    semantic_pred = F.interpolate(semantic_pred[None, :, :, :].float(), OUT_RES, mode="nearest")[0].int()

    return semantic_pred


def save_prediction(path: str, filename: str, prediction):
    prediction = prediction.reshape(prediction.shape[1], prediction.shape[2])
    prediction = np.array(prediction.cpu()).astype(np.uint8)
    image = Image.fromarray(prediction)
    if not DRY_RUN:
        image.save(os.path.join(path, filename))


def convert_sequence(input_dir, seq: str, model, device, config, transforms):
    # image_00
    convert_folder(
        folder_path=os.path.join(input_dir, seq, 'image_00', 'data_rect'),
        output_path=os.path.join(config.output_dir, seq, 'image_00', 'data_192x640'),
        model=model,
        device=device,
        transforms=transforms
    )

    # image_01
    convert_folder(
        folder_path=os.path.join(input_dir, seq, 'image_01', 'data_rect'),
        output_path=os.path.join(config.output_dir, seq, 'image_01', 'data_192x640'),
        model=model,
        device=device,
        transforms=transforms
    )

    # image_02
    convert_folder(
        folder_path=os.path.join(input_dir, seq, 'image_02', 'data_192x640_0x-15'),
        output_path=os.path.join(config.output_dir, seq, 'image_02', 'data_192x640_0x-15'),
        model=model,
        device=device,
        transforms=transforms
    )

    # image_03
    convert_folder(
        folder_path=os.path.join(input_dir, seq, 'image_03', 'data_192x640_0x-15'),
        output_path=os.path.join(config.output_dir, seq, 'image_03', 'data_192x640_0x-15'),
        model=model,
        device=device,
        transforms=transforms
    )


def convert_folder(folder_path, output_path, model, device, transforms):
    logging.info(f"Converting folder at {folder_path}")

    if not os.path.exists(output_path):
        logging.info(f"Output directory {output_path} does not exist and has to be created!")
        os.makedirs(output_path)

    for filename in tqdm(sorted(os.listdir(folder_path))):
        _pred = predict_image(os.path.join(folder_path, filename), model=model, device=device, transforms=transforms)
        save_prediction(output_path, filename, _pred)

    logging.info("Conversion finished")


def main():
    args = parse_args()

    model, device, config, transforms = setup_model(args)

    with torch.no_grad():
        for seq in os.listdir(args.input_dir):
            convert_sequence(args.input_dir, seq, model, device, config, transforms)


if __name__ == '__main__':
    main()
