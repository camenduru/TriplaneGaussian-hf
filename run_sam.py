from utils import image_preprocess, pred_bbox, sam_init, sam_out_nosave
import os
from PIL import Image
import argparse

SAM_CKPT_PATH = "code/checkpoints/sam_vit_h_4b8939.pth"

def resize_image(input_raw, size):
    w, h = input_raw.size
    ratio = size / max(w, h)
    resized_w = int(w * ratio)
    resized_h = int(h * ratio)
    return input_raw.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

if __name__ == "__main__":
    # load SAM checkpoint
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    sam_predictor = sam_init(SAM_CKPT_PATH, gpu)
    print("load sam ckpt done.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()

    input_raw = Image.open(args.image_path)
    # input_raw.thumbnail([512, 512], Image.Resampling.LANCZOS)
    input_raw = resize_image(input_raw, 512)
    image_sam = sam_out_nosave(
        sam_predictor, input_raw.convert("RGB"), pred_bbox(input_raw)
    )

    # save_path = os.path.join(args.save_path, "input_rgba.png")
    image_preprocess(image_sam, args.save_path, lower_contrast=False, rescale=True)
