import os, sys

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

# If you have multiple GPUs, you can set the GPU to use here.
# The default is to use the first GPU, which is usually GPU 0.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import os
import copy
from glob import glob
from tqdm import tqdm
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict


# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

import PIL
import requests
import torch
from io import BytesIO

from huggingface_hub import hf_hub_download
from supervision.draw.color import Color, ColorPalette
self_color= ColorPalette.default()

### Load Grounding DINO model
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

# Use this command for evaluate the Grounding DINO model
# Or you can download the model by yourself
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

TEXT_PROMPTs = ["car", "bus","motorcycle", "bicycle", "human",
                "traffic light", "guard rail",  "pole", "sign",
                ][::-1]#"road", "ground"]
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.25
device = "cuda:1"
device ='cpu'
#device ='cuda:0'

### load SAM model
sam_checkpoint = '../segment-anything/test_SAM/sam_vit_h_4b8939.pth'

sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)

def show_mask_as_set_color(mask, image,Color):
    
    color = np.array([Color.r/255, Color.g/255, Color.b/255, 0.8])#np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

### General setting
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

data_name = "rosbag2_2023_03_03-14_28_12_10"
camera_name = 'c2'

data_name_list = [
    # "rosbag2_2023_03_03-13_40_02_0", #573
#                   "rosbag2_2023_03_03-13_40_02_1", #586
#                   "rosbag2_2023_03_03-13_40_02_2", #579
#                   "rosbag2_2023_03_03-13_40_02_3", #576
#                   "rosbag2_2023_03_03-13_40_02_4", #588
#                   "rosbag2_2023_03_03-13_40_02_5", #590
#                   "rosbag2_2023_03_03-13_40_02_6", #592
#                   "rosbag2_2023_03_03-13_40_02_7", #585
#                   "rosbag2_2023_03_03-13_40_02_8", #185
                  
                  "rosbag2_2023_03_03-13_54_05_0", #580
                  "rosbag2_2023_03_03-13_54_05_1", #579
                  "rosbag2_2023_03_03-13_54_05_2", #585
                  "rosbag2_2023_03_03-13_54_05_3", #593
                  "rosbag2_2023_03_03-13_54_05_4", #591
                  "rosbag2_2023_03_03-13_54_05_5", #586
                  "rosbag2_2023_03_03-13_54_05_6", #590
                  "rosbag2_2023_03_03-13_54_05_7", #589
                  "rosbag2_2023_03_03-13_54_05_8", #587
                  "rosbag2_2023_03_03-13_54_05_9",  #587
                  "rosbag2_2023_03_03-13_54_05_10", #584
                  "rosbag2_2023_03_03-13_54_05_11", #579
                  "rosbag2_2023_03_03-13_54_05_12", #426
                  
                  "rosbag2_2023_03_03-14_46_14_0", #582
                  "rosbag2_2023_03_03-14_46_14_1", #594
                  "rosbag2_2023_03_03-14_46_14_2", #588
                  "rosbag2_2023_03_03-14_46_14_3", #589
                  "rosbag2_2023_03_03-14_46_14_4", #588
                  "rosbag2_2023_03_03-14_46_14_5", #588
                  "rosbag2_2023_03_03-14_46_14_6", #586
                  "rosbag2_2023_03_03-14_46_14_7", #585
                  "rosbag2_2023_03_03-14_46_14_8", #575
                  "rosbag2_2023_03_03-14_46_14_9", #580
                  "rosbag2_2023_03_03-14_46_14_10", #583
                  "rosbag2_2023_03_03-14_46_14_11", #426
                  
                  
                  
                  
                  ]
camera_name_list = ['c1', 'c2']
camera_name_list = ['c1']

for data_name in data_name_list:
    for camera_name in camera_name_list:
        result_bbox_dir = f'results/bbox/png/{data_name}/{camera_name}'
        result_bbox_concat_dir = f'results/bbox_concat/png/{data_name}/{camera_name}'
        os.makedirs(result_bbox_dir, exist_ok=True)
        os.makedirs(result_bbox_concat_dir, exist_ok=True)

        result_dir = f'results/ann/png/{data_name}/{camera_name}'
        result_concat_dir = f'results/ann_concat/png/{data_name}/{camera_name}'
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(result_concat_dir, exist_ok=True)



        # image path

        image_paths = sorted(glob(f'../../data/odaiba_data/results/{data_name}/{camera_name}/image_raw/compressed/*.png'))
        assert len(image_paths) >0
        result_dir = f'results/ann/png/{data_name}/{camera_name}'
        result_concat_dir = f'results/concat/png/{data_name}/{camera_name}'
        #result_jpg_dir = f'results/ann/jpg/{data_name}/{camera_name}'
        #result_jpg_concat_dir = f'results/concat/jpg/{data_name}/{camera_name}'
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(result_concat_dir, exist_ok=True)
        #os.makedirs(result_jpg_dir, exist_ok=True)
        #os.makedirs(result_jpg_concat_dir, exist_ok=True)


        for i, image_p in tqdm(enumerate(image_paths)):
            
            local_image_path = image_p
            
            image_source, image = load_image(local_image_path)
            
            
            image_base = image_source
            # set image
            sam_predictor.set_image(image_source)
            

            for idx, TEXT_PROMPT in enumerate(TEXT_PROMPTs):
                
                boxes, logits, phrases = predict(
                    model=groundingdino_model, 
                    image=image, 
                    caption=TEXT_PROMPT, 
                    box_threshold=BOX_TRESHOLD, 
                    text_threshold=TEXT_TRESHOLD, device='cpu'
                )

                annotated_frame = annotate(image_source=image_base, boxes=boxes, logits=logits, phrases=phrases)
                annotated_frame = annotated_frame[...,::-1]
                
                
                # box: normalized box xywh -> unnormalized xyxy
                H, W, _ = image_source.shape
                boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
                
                transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
                if len(transformed_boxes) > 0:
                    masks, _, _ = sam_predictor.predict_torch(
                                point_coords = None,
                                point_labels = None,
                                boxes = transformed_boxes,
                                multimask_output = False,
                            )
                    print(len(masks))
                    for j in range(len(masks)):
                        annotated_frame =  show_mask_as_set_color(masks[j][0].cpu(), annotated_frame,self_color.colors[idx] )
                    annotated_frame_with_mask = annotated_frame
                    #annotated_frame_with_mask = show_mask(masks[0][0], annotated_frame)
                    image_base = annotated_frame_with_mask
            annotated_frame = image_base # RGBA
            #print(image_p)
            
            
            save_name = f'{i:05d}.png'
            pil_annotated= Image.fromarray(annotated_frame)
            pil_annotated.save(os.path.join(result_dir, save_name))
            
            pil_source = Image.fromarray(image_source)
            
            pil_concat = get_concat_h(pil_source, pil_annotated)
            pil_concat.save(os.path.join(result_concat_dir, save_name))
            

            
            
        command = f"ffmpeg -r 30 -i {result_dir}/%05d.png -vcodec libx264 -pix_fmt yuv420p -r 30 -y results/ann/png/{data_name}/{camera_name}.mp4"
        subprocess.run(command,shell=True)
            
        command = f"ffmpeg -r 30 -i {result_concat_dir}/%05d.png -vcodec libx264 -pix_fmt yuv420p -r 30 -y results/concat/png/{data_name}/{camera_name}.mp4"
        subprocess.run(command,shell=True)
            