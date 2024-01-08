import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from rembg import remove
from segment_anything import SamPredictor, sam_model_registry
import urllib.request
from tqdm import tqdm


def sam_init(sam_checkpoint, device_id=0):
    # sam_checkpoint = os.path.join(os.path.dirname(__file__), "./sam_vit_h_4b8939.pth")
    model_type = "vit_h"

    device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def sam_out_nosave(predictor, input_image, *bbox_sliders):
    bbox = np.array(bbox_sliders)
    image = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image)

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(
        box=bbox, multimask_output=True
    )

    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = (
        masks_bbox[-1].astype(np.uint8) * 255
    )  # np.argmax(scores_bbox)
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode="RGBA")


# contrast correction, rescale and recenter
def image_preprocess(input_image, save_path, lower_contrast=True, rescale=True):
    image_arr = np.array(input_image)
    in_w, in_h = image_arr.shape[:2]

    if lower_contrast:
        alpha = 0.8  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)
        # Apply the contrast adjustment
        image_arr = cv2.convertScaleAbs(image_arr, alpha=alpha, beta=beta)
        image_arr[image_arr[..., -1] > 200, -1] = 255

    ret, mask = cv2.threshold(
        np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY
    )
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    ratio = 0.75
    if rescale:
        side_len = int(max_size / ratio)
    else:
        side_len = in_w
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2
    padded_image[
        center - h // 2 : center - h // 2 + h, center - w // 2 : center - w // 2 + w
    ] = image_arr[y : y + h, x : x + w]
    rgba = Image.fromarray(padded_image).resize((256, 256), Image.LANCZOS)
    rgba.save(save_path)

    # rgba_arr = np.array(rgba) / 255.0
    # rgb = rgba_arr[...,:3] * rgba_arr[...,-1:] + (1 - rgba_arr[...,-1:])
    # return Image.fromarray((rgb * 255).astype(np.uint8))


def pred_bbox(image):
    image_nobg = remove(image.convert("RGBA"), alpha_matting=True)
    alpha = np.asarray(image_nobg)[:, :, -1]
    x_nonzero = np.nonzero(alpha.sum(axis=0))
    y_nonzero = np.nonzero(alpha.sum(axis=1))
    x_min = int(x_nonzero[0].min())
    y_min = int(y_nonzero[0].min())
    x_max = int(x_nonzero[0].max())
    y_max = int(y_nonzero[0].max())
    return x_min, y_min, x_max, y_max

# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars, *args, **kwargs):
        if isinstance(vars, list):
            return [wrapper(x, *args, **kwargs) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x, *args, **kwargs) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v, *args, **kwargs) for k, v in vars.items()}
        else:
            return func(vars, *args, **kwargs)

    return wrapper

@make_recursive_func
def todevice(vars, device="cuda"):
    if isinstance(vars, torch.Tensor):
        return vars.to(device)
    elif isinstance(vars, str):
        return vars
    elif isinstance(vars, bool):
        return vars
    elif isinstance(vars, float):
        return vars
    elif isinstance(vars, int):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))

def download_checkpoint(url, save_path):
    try:
        with urllib.request.urlopen(url) as response, open(save_path, 'wb') as file:
            file_size = int(response.info().get('Content-Length', -1))
            chunk_size = 8192
            num_chunks = file_size // chunk_size if file_size > chunk_size else 1

            with tqdm(total=file_size, unit='B', unit_scale=True, desc='Downloading', ncols=100) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), b''):
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"Checkpoint downloaded and saved to: {save_path}")
    except Exception as e:
        print(f"Error downloading checkpoint: {e}")
