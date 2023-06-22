#!/bin/bash
scan="scene0616_00"

# image sharpening
python preprocess/img_preprocess.py --scene ${scan}

# pred normal
cd snucode
python test2.py --pretrained scannet_neuris_retrain --architecture BN --imgs_dir ../dataset/indoor/${scan}/image_process
python gen_npz.py --dir_neus ../dataset/indoor/${scan}