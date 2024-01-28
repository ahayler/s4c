#!/bin/bash

echo "----------------------- Downloading pretrained BehindTheScenes backbone -----------------------"

cp_link="https://cvg.cit.tum.de/webshare/g/behindthescenes/kitti-360/training-checkpoint.pt"
cp_download_path="out/kitti_360/backbone/training-checkpoint.pt"

basedir=$(dirname $0)
outdir=$(dirname $cp_download_path)

cd $basedir || exit
echo Operating in \"$(pwd)\".
echo Creating directories.
mkdir -p $outdir
echo Downloading checkpoint from \"$cp_link\" to \"$cp_download_path\".
wget -O $cp_download_path $cp_link