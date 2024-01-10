import argparse
import os
import glob
import torch
from PIL import Image
from copy import deepcopy
import sys
import tempfile
import subprocess
from huggingface_hub import snapshot_download
from functools import partial

LOCAL_CODE = os.environ.get("LOCAL_CODE", "1") == "1"
CACHE_EXAMPLES = os.environ.get("CACHE_EXAMPLES", "1") == "1"
SAM_LOCAL = os.environ.get("SAM_LOCAL", "1") == "1"
AUTH = ("admin", os.environ["PASSWD"]) if "PASSWD" in os.environ else None
DEFAULT_CAM_DIST = 1.9

code_dir = snapshot_download("zouzx/TriplaneGaussian", local_dir="./code", token=os.environ["HF_TOKEN"]) if not LOCAL_CODE else "./code"

sys.path.append(code_dir)

if not LOCAL_CODE:
    subprocess.run(["pip", "install", "--upgrade", "gradio==4.12.0"])

import gradio as gr
print("gr version: ", gr.__version__)

from utils import image_preprocess, pred_bbox, sam_init, sam_out_nosave, todevice
from gradio_splatting.backend.gradio_model3dgs import Model3DGS
import tgs
from tgs.utils.config import ExperimentConfig, load_config
from tgs.systems.infer import TGS

SAM_CKPT_PATH = "code/checkpoints/sam_vit_h_4b8939.pth"
MODEL_CKPT_PATH = "code/checkpoints/tgs_lvis_100v_rel.ckpt"
CONFIG = "code/configs/single-rel.yaml"
EXP_ROOT_DIR = "./outputs-gradio"

os.makedirs(EXP_ROOT_DIR, exist_ok=True)

gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
device = "cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu"

print("device: ", device)

# init system
base_cfg: ExperimentConfig
base_cfg = load_config(CONFIG, cli_args=[], n_gpus=1)
base_cfg.system.weights = MODEL_CKPT_PATH
system = TGS(cfg=base_cfg.system).to(device)
print("load model ckpt done.")

HEADER = """
# Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D Reconstruction with Transformers

<div>
<a style="display: inline-block;" href="https://arxiv.org/abs/2312.09147"><img src="https://img.shields.io/badge/arxiv-2312.09147-B31B1B.svg"></a>
</div>

TGS enables fast reconstruction from single-view image in a few seconds based on a hybrid Triplane-Gaussian 3D representation.

This model is trained on Objaverse-LVIS (**~45K** synthetic objects) only. And note that we normalize the input camera pose to a pre-set viewpoint during training stage following LRM, rather than directly using camera pose of input camera as implemented in our original paper.

**Tips:**
1. If you find the result is unsatisfied, please try to change the camera distance. It perhaps improves the results.
2. Please wait until the completion of the reconstruction of the previous model before proceeding with the next one, otherwise, it may cause bug. We will fix it soon.
"""

def assert_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image selected or uploaded!")

def resize_image(input_raw, size):
    w, h = input_raw.size
    ratio = size / max(w, h)
    resized_w = int(w * ratio)
    resized_h = int(h * ratio)
    return input_raw.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

def preprocess(input_raw, save_path, sam_predictor=None):
    # if not preprocess:
    #     print("No preprocess")
    #     # return image_path
    image_path = os.path.join(save_path, "input_raw.png")
    save_path = os.path.join(save_path, "seg_rgba.png")
    if SAM_LOCAL and sam_predictor is not None:
        # input_raw = Image.open(image_path)
        # input_raw.thumbnail([512, 512], Image.Resampling.LANCZOS)
        input_raw = resize_image(input_raw, 512)
        print("image size:", input_raw.size)
        image_sam = sam_out_nosave(
            sam_predictor, input_raw.convert("RGB"), pred_bbox(input_raw)
        )
        image_preprocess(image_sam, save_path, lower_contrast=False, rescale=True)
    else:
        input_raw.save(image_path)
        subprocess.run([f"python run_sam.py --image_path {image_path} --save_path {save_path}"], shell=True)

    print("image raw path = ", image_path, "image save path =", save_path)
    return save_path

def init_trial_dir():
    trial_dir = tempfile.TemporaryDirectory(dir=EXP_ROOT_DIR).name
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir

@torch.no_grad()
def infer(image_path: str,
          cam_dist: float,
          save_path: str,
          only_3dgs: bool = False):
    data_cfg = deepcopy(base_cfg.data)
    data_cfg.only_3dgs = only_3dgs
    data_cfg.cond_camera_distance = cam_dist
    data_cfg.eval_camera_distance = cam_dist
    data_cfg.image_list = [image_path]
    dm = tgs.find(base_cfg.data_cls)(data_cfg)

    dm.setup()
    for batch_idx, batch in enumerate(dm.test_dataloader()):
        batch = todevice(batch, device)
        system.test_step(save_path, batch, batch_idx, save_3dgs=only_3dgs)
    if not only_3dgs:
        system.on_test_epoch_end(save_path)

def run(image_path: str,
        cam_dist: float,
        save_path: str):
    infer(image_path, cam_dist, save_path, only_3dgs=True)
    gs = glob.glob(os.path.join(save_path, "*.ply"))[0]
    # print("save gs", gs)
    return gs

def run_video(image_path: str,
            cam_dist: float,
            save_path: str):
    infer(image_path, cam_dist, save_path)
    video = glob.glob(os.path.join(save_path, "*.mp4"))[0]
    # print("save video", video)
    return video

def run_example(image_path, sam_predictor=None):
    save_path = init_trial_dir()
    seg_image_path = preprocess(image_path, save_path, sam_predictor)
    gs = run(seg_image_path, DEFAULT_CAM_DIST, save_path)
    video = run_video(seg_image_path, DEFAULT_CAM_DIST, save_path)
    return seg_image_path, gs, video

def launch(port):
    if SAM_LOCAL:
        sam_predictor = sam_init(SAM_CKPT_PATH, gpu)
        print("load sam ckpt done.")

    with gr.Blocks(
        title="TGS - Demo"
    ) as demo:
        with gr.Row(variant='panel'):
            gr.Markdown(HEADER)
    
        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                input_image = gr.Image(value=None, image_mode="RGB", width=512, height=512, type="pil", sources="upload", label="Input Image")
                gr.Markdown(
                    """
                    **Camera distance** denotes the distance between camera center and scene center.
                    If you find the 3D model appears flattened, you can increase it. Conversely, if the 3D model appears thick, you can decrease it.
                    """
                )
                camera_dist_slider = gr.Slider(1.0, 4.0, value=DEFAULT_CAM_DIST, step=0.1, label="Camera Distance")
                # preprocess_ckb = gr.Checkbox(value=True, label="Remove background")
                img_run_btn = gr.Button("Reconstruction", variant="primary")

            with gr.Column(scale=1):
                with gr.Row(variant='panel'):
                    seg_image = gr.Image(value=None, width="auto", type="filepath", image_mode="RGBA", label="Segmented Image", interactive=False)
                    output_video = gr.Video(value=None, width="auto", label="Rendered Video", autoplay=True)
                output_3dgs = Model3DGS(value=None, label="3D Model")
        
        with gr.Row(variant="panel"):
            gr.Examples(
                examples=[
                    "example_images/green_parrot.webp",
                    "example_images/rusty_gameboy.webp",
                    "example_images/a_pikachu_with_smily_face.webp",
                    "example_images/an_otter_wearing_sunglasses.webp",
                    "example_images/lumberjack_axe.webp",
                    "example_images/medieval_shield.webp",
                    "example_images/a_cat_dressed_as_the_pope.webp",
                    "example_images/a_cute_little_frog_comicbook_style.webp",
                    "example_images/a_purple_winter_jacket.webp",
                    "example_images/MP5,_high_quality,_ultra_realistic.webp",
                    "example_images/retro_pc_photorealistic_high_detailed.webp",
                    "example_images/stratocaster_guitar_pixar_style.webp"
                ],
                inputs=[input_image],
                outputs=[seg_image, output_3dgs, output_video],
                cache_examples=CACHE_EXAMPLES,
                fn=partial(run_example, sam_predictor=sam_predictor),
                label="Examples",
                examples_per_page=40
            )

        trial_dir = gr.State()
        img_run_btn.click(
            fn=assert_input_image,
            inputs=[input_image],
            # queue=False
        ).success(
            fn=init_trial_dir,
            outputs=[trial_dir],
            # queue=False
        ).success(
            fn=partial(preprocess, sam_predictor=sam_predictor),
            inputs=[input_image, trial_dir],
            outputs=[seg_image],
        ).success(fn=run,
                inputs=[seg_image, camera_dist_slider, trial_dir],
                outputs=[output_3dgs],
        ).success(fn=run_video,
                inputs=[seg_image, camera_dist_slider, trial_dir],
                outputs=[output_video])

        launch_args = {"server_port": port}
        demo.queue(max_size=20)
        demo.launch(auth=AUTH, **launch_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, extra = parser.parse_known_args()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch(args.port)