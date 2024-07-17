import os
import torch
import diffusers
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image
import numpy as np

import sys
sys.path.append("./")
from mvdream_control.camera_utils import get_camera
from threestudio.data.dataset_objaverse_sketch_np import Warper, sketch_warp_new
from pathlib import Path

import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(description="Construct coarse 3D meshes")
    
    parser.add_argument('--sketch_path', type=str, required=True, default=None, help='Path of the sketch path')
    parser.add_argument('--prompt', type=str, required=True, default=None, help='the text prompt for the depth generation')
    parser.add_argument('--output_image', type=str, required=True, default=None, help='Path of the output depth image')
    parser.add_argument('--output_np', type=str, required=True, default=None, help='Path of the output depth numpy array')
    parser.add_argument('--output_warp_path', type=str, required=True, default=None, help='Path of the output warp depth')
    parser.add_argument('--seed', type=int, required=False, default=0, help='The generation seed')
    parser.add_argument('--depth_bias', type=float, required=False, default=0.0, help='A global bias for the depth maps')
    parser.add_argument('--azimuth_source', type=float, required=False, default=0.0, help='the azimuth angle for sketch')
    parser.add_argument('--azimuth_target', type=float, required=False, default=45.0, help='the azimuth angle for warp view')
    parser.add_argument('--elevation_source', type=float, required=False, default=15.0, help='the elevation angle for sketch')
    parser.add_argument('--elevation_target', type=float, required=False, default=15.0, help='the elevation angle for warp view')
    
    args = parser.parse_args()
    
    depth_bias = args.depth_bias
    azimuth_source = args.azimuth_source
    azimuth_target = args.azimuth_target
    elevation_source = args.elevation_source
    elevation_target = args.elevation_target
    
    # Define the pretrained model path
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    unet_path = "./models/sketch-to-depth"
    controlnet_path = "lllyasviel/control_v11p_sd15_lineart"
    
    # Load the pretrained models
    tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        unet_path, subfolder="unet_ema"
    )
    controlnet = ControlNetModel.from_pretrained(controlnet_path)

    # build the running pipeline
    weight_dtype = torch.float16
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=weight_dtype,
        )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to('cuda')

    resolution = 256
    validation_image = Image.open(args.sketch_path).convert("RGB")        
    control_image = validation_image.resize((resolution, resolution))
    control_image = np.array(control_image)
    control_image = control_image[None, :].astype(np.float32) / 255.0
    control_image = control_image.transpose(0, 3, 1, 2)
    control_image = torch.from_numpy(control_image)
    control_image = 1.0 - control_image

    seed = args.seed
    
    # predict depth maps
    generator = torch.Generator(device='cuda').manual_seed(seed)
    with torch.no_grad():
        image = pipeline(
            args.prompt, control_image, num_inference_steps=20, generator=generator
        ).images[0]

    image.save(args.output_image)
    
    # define warp
    warper = Warper()
    
    # define camera parameters
    camera = get_camera(4, elevation=elevation_target, azimuth_start=float(azimuth_target))
    camera_source = get_camera(1, elevation=elevation_source, azimuth_start=float(azimuth_source))
    camera = np.concatenate((camera_source,camera), axis=0)
    camera = torch.tensor(camera)

    azimuth_list = [azimuth_source, (azimuth_target)%360, (azimuth_target+90)%360, (azimuth_target+180)%360, (azimuth_target+270)%360]
    elevation_list = [elevation_source, elevation_target, elevation_target, elevation_target, elevation_target]

    depth_input = np.array(image) / 255.0
    depth_input = np.mean(depth_input, axis=2)
    depth_input = depth_input + depth_bias
    bk = depth_input > (0.95 + depth_bias)
    depth_input[bk] = 1.0

    sketch_warper_input = warper.read_image(Path(args.sketch_path))

    azimuth = azimuth_list[1]
    elevation = elevation_list[1]
    
    mask_img = np.ones(sketch_warper_input.shape) * 255 * (depth_input.reshape(256,256,1) != 1)
    warp_test = sketch_warp_new(warper, mask_img, sketch_warper_input, depth_input, np.deg2rad(elevation_source), np.deg2rad(elevation), \
                                          np.deg2rad(azimuth_list[0]), np.deg2rad(azimuth))
    warp_test = warp_test.reshape(256,256,1).repeat(3,2)

    white_img = np.ones(sketch_warper_input.shape) * 255
    warp_white_img = sketch_warp_new(warper, mask_img, white_img, depth_input, np.deg2rad(elevation_source), np.deg2rad(elevation), \
                                          np.deg2rad(azimuth_list[0]), np.deg2rad(azimuth))
    warp_white_img = warp_white_img.reshape(256,256,1).repeat(3,2)

    kernel = np.ones((3,3),np.uint8)
    warp_white_img_process = cv2.morphologyEx(warp_white_img, cv2.MORPH_OPEN, kernel)
    warp_white_img_process = cv2.morphologyEx(warp_white_img_process, cv2.MORPH_CLOSE, kernel)

    index = warp_white_img_process != warp_white_img
    warp_test[index] = warp_white_img_process[index]

    warp_test_pil = Image.fromarray(warp_test)
    warp_test_pil.save(args.output_warp_path)
    
    np.save(args.output_np, depth_input)

if __name__ == "__main__":
    main()

# python depth_predict.py --sketch_path ./asset/golden_fish.png --prompt "a 3D model of realistic golden fish" --output_image ./assert/golden_fish_depth.png --output_np ./assert/golden_fish_depth.npy --seed 69646
