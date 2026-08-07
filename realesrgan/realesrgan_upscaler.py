#!/usr/bin/python
'''RealESRGAN Upscaler node.'''
# pylint: disable=invalid-name
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-arguments
# pylint: disable=unused-variable
# pylint: disable=bare-except
# pylint: disable=too-many-locals
# pylint: disable=line-too-long
# pylint: disable=no-member

# Import the Python modules.
import os
import gc
import warnings
import pathlib

# Set some module strings.
__author__ = "zentrocdot"
__copyright__ = "© Copyright 2025, zentrocdot"
__version__ = "0.0.1.1"

# Import the third party Python modules.
from PIL import Image
import cv2
import numpy as np
import torch
import requests
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Disable future warning.
warnings.filterwarnings("ignore", category=FutureWarning)

# Create a context manager.
class ClearCache:
    '''Clear cache class.'''
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        torch.cuda.empty_cache()

# Get the number of GPUs.
NUM_GPUS = torch.cuda.device_count()

# Initialse the GPU string.
GPU_STR = ""

# Create the GPU string.
for i in range(NUM_GPUS):
    VRAM = torch.cuda.get_device_properties(i).total_memory
    VRAM = str(int(VRAM / 1000**3)) + " GB"
    info = torch.cuda.get_device_name(i)
    print(i, info)
    GPU_STR = GPU_STR + str(i) + " " + info + " " + VRAM + "\n"
GPU_STR = GPU_STR.lstrip().rstrip()

# Create list with the GPUs numbers.
GPU_LIST = [str(i) for i in range(NUM_GPUS)]

# Set some paths.
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
PARENT_PATH = SCRIPT_PATH.parent.absolute()
MODELS_PATH = ''.join([str(PARENT_PATH), "/models"])

# Make sure the models directory exists (Fixed: Prevents FileNotFoundError)
os.makedirs(MODELS_PATH, exist_ok=True)

# Set file paths.
MOD_DIR = {'RealESRGAN_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
           'RealESRGAN_x2plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
           'RealESRNet_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
           'ESRGAN_SRx4_DF2KOST_official-ff704c30.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'}

# Download models.
for i, (k, v) in enumerate(MOD_DIR.items()):
    model_path = '/'.join([str(MODELS_PATH), k])
    model_file = pathlib.Path(model_path)
    print(model_file)
    if not model_file.is_file():
        response = requests.get(v, timeout=30)
        if response.status_code == 200:
            with open(model_path, 'wb') as file:
                file.write(response.content)
            print('File download succeeded!')
        else:
            print('File download failed!')

# Read models in dir into list.
MODS = []
for f in os.listdir(MODELS_PATH):
    if f.endswith('.pth'):
        MODS.append(f)

# -------------------------------
# Convert Tensor to PIL function.
# -------------------------------
def tensor2pil(image):
    '''Tensor to PIL image.'''
    # Return a PIL image.
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# ---------------------------------
# Convert Numpy to Tensor function.
# ---------------------------------
def numpy2tensor(image):
    '''Numpy image to Tensor.'''
    # Return a tensor.
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# ----------------------------
# RealESRGAN Upscaler function
# ----------------------------
def upscaler(input_img, outscale, gpu_id, tile, fp_fmt, denoise, netscale, tile_pad, pre_pad, models):
    """Inference upscaler using Real-ESRGAN.
    """
    ERROR = None
    MODEL_PATH = '/'.join([str(MODELS_PATH), models])
    # Convert PIL image to Numpy array.
    image = np.array(input_img)
    # Set up the network for the RealESRGAN model. Set scale to netscale.
    # Set the netscale (4) to the depending one of the model (x4).
    scale = netscale
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    # Set dni_weight to control the denoise strength.
    if denoise > 0:
        dni_weight = [denoise, 1 - denoise]
    else:
        dni_weight = None
    # Define the upscaler upsampler.
    with ClearCache():
        try:
            upsampler = RealESRGANer(
                scale=netscale,
                model_path=MODEL_PATH,
                dni_weight=dni_weight,
                model=model,
                tile=tile,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                half=(fp_fmt == "fp16"),
                gpu_id = int(gpu_id)
            )
        except:
            ERROR = "💥 Oops, something went wrong!\n" + \
                    "Check whether netscale fits the model!"
            return None, ERROR
    # Determine the output image.
    try:
        outimg, _ = upsampler.enhance(image, outscale=outscale)
    except RuntimeError as error:
        torch.cuda.empty_cache()
        outimg = None
        ERROR = "An serious error occurred 💥. " + \
                "CUDA out of memory. Try to set tile size to a larger/smaller number."
    except UnboundLocalError as error:
        torch.cuda.empty_cache()
        outimg = None
        ERROR = "An serious error occurred 💥. " + \
                "Image size problem occurs. Tile size problem occurs. Check terminal."
    # Return the upscaled image.
    return outimg, ERROR

# ++++++++++++++++++++++++
# Class RealEsrganUpscaler
# ++++++++++++++++++++++++
class RealEsrganUpscaler:
    '''real esrgan upscaler.'''

    @classmethod
    def INPUT_TYPES(cls):
        '''Define the node input types.'''
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1024.0, "step": 0.1}),
                "tile_number": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64}),
                "tile_pad": ("INT", {"default": 10, "min": 0, "max": 1024, "step": 1}),
                "pre_pad": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "fp_format": (["fp16", "fp32"], {}),
                "denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gpu_id": (GPU_LIST, {}),
                "netscale": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "models": (MODS, {}),
            }
        }

    # Set the ComfyUI related variables.
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING", "STRING",)
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height", "config_str", "error_str",)
    FUNCTION = "realesrgan_upscaler"
    CATEGORY = "💊 RealEsrganUpscaler"
    DESCRIPTION = "Upscaling using RealESRGAN."
    OUTPUT_NODE = True

    def realesrgan_upscaler(self, image, gpu_id, scale_factor, tile_number,
                            fp_format, models, denoise, netscale, tile_pad,
                            pre_pad):
        '''RealESRGAN upscaler.'''
        # Set value for paranoia reason.
        gpu_id = int(gpu_id)
        # Create a PIL image.
        img_input = tensor2pil(image)
        # Copy image.
        imgNEW = img_input.copy()
        # Upscale image.
        imgNEW, ERROR = upscaler(imgNEW, scale_factor, gpu_id, tile_number,
                                 fp_format, denoise, netscale, tile_pad,
                                 pre_pad, models)
        # Check if ERROR or imgNEW is None.
        if ERROR is not None or imgNEW is None:
            # Set rows and cols.
            n,m = 512,512
            # Create an empty image.
            imgNEW = np.zeros([n,m,3], dtype=np.uint8)
            # Add text to the image
            text = "Oops, something went wrong!"
            position = (20, 256)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 0, 255)
            thickness = 2
            cv2.putText(imgNEW, text, position, font, font_scale, color, thickness)
        # Get width and height of the new image.
        (height, width, channels) = imgNEW.shape
        # Create a mask.
        maskImage = np.zeros((height, width, channels), np.uint8)
        mask = numpy2tensor(maskImage)
        mask = mask[:, :, :, 1]
        # Create tensors from PIL.
        image_out = numpy2tensor(imgNEW)
        # Return the upscaled image.
        return (image_out, mask, width, height, GPU_STR, ERROR,)
