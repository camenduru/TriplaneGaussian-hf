import gradio as gr
import argparse
import os
import glob
import torch
from PIL import Image
from copy import deepcopy
import sys
import tempfile
from huggingface_hub import snapshot_download

LOCAL_CODE = os.environ.get("LOCAL_CODE", "1") == "1"
AUTH = ("admin", os.environ["PASSWD"]) if "PASSWD" in os.environ else None

code_dir = snapshot_download("zouzx/TriplaneGaussian", local_dir="./code", token=os.environ["HF_TOKEN"]) if not LOCAL_CODE else "./code"

sys.path.append(code_dir)

from utils import image_preprocess, pred_bbox, sam_init, sam_out_nosave, todevice
from gradio_splatting.backend.gradio_model3dgs import Model3DGS
import tgs
from tgs.utils.config import ExperimentConfig, load_config
from tgs.systems.infer import TGS

SAM_CKPT_PATH = "code/checkpoints/sam_vit_h_4b8939.pth"
MODEL_CKPT_PATH = "code/checkpoints/tgs_lvis_100v_rel.ckpt"
CONFIG = "code/configs/single-rel.yaml"
EXP_ROOT_DIR = "./outputs-gradio"

gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
device = "cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu"

print("device: ", device)

# load SAM checkpoint
sam_predictor = sam_init(SAM_CKPT_PATH, gpu)
print("load sam ckpt done.")

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

This model is trained on Objaverse-LVIS (~40K synthetic objects) only. And note that we normalize the input camera pose to a pre-set viewpoint during training stage following LRM, rather than directly using camera pose of input camera as implemented in our original paper.
"""

def preprocess(image_path, save_path=None, lower_contrast=False):
    input_raw = Image.open(image_path)

    input_raw.thumbnail([512, 512], Image.Resampling.LANCZOS)
    image_sam = sam_out_nosave(
        sam_predictor, input_raw.convert("RGB"), pred_bbox(input_raw)
    )

    if save_path is None:
        save_path, ext = os.path.splitext(image_path)
        save_path = save_path + "_rgba.png"
    image_preprocess(image_sam, save_path, lower_contrast=lower_contrast, rescale=True)

    return save_path

def init_trial_dir():
    if not os.path.exists(EXP_ROOT_DIR):
        os.makedirs(EXP_ROOT_DIR, exist_ok=True)
    trial_dir = tempfile.TemporaryDirectory(dir=EXP_ROOT_DIR).name
    system.set_save_dir(trial_dir)
    return trial_dir

@torch.no_grad()
def infer(image_path: str,
          cam_dist: float,
          fovy_deg: float,
          only_3dgs: bool = False):
    data_cfg = deepcopy(base_cfg.data)
    data_cfg.only_3dgs = only_3dgs
    data_cfg.cond_fovy_deg = fovy_deg
    data_cfg.cond_camera_distance = cam_dist
    data_cfg.image_list = [image_path]
    dm = tgs.find(base_cfg.data_cls)(data_cfg)

    dm.setup()
    for batch_idx, batch in enumerate(dm.test_dataloader()):
        batch = todevice(batch, device)
        system.test_step(batch, batch_idx, save_3dgs=only_3dgs)
    if not only_3dgs:
        system.on_test_epoch_end()

def run(image_path: str,
        cam_dist: float,
        fov_degree: float):
    infer(image_path, cam_dist, fov_degree, only_3dgs=True)
    save_path = system.get_save_dir()
    gs = glob.glob(os.path.join(save_path, "*.ply"))[0]
    return gs

def run_video(image_path: str,
            cam_dist: float,
            fov_degree: float):
    infer(image_path, cam_dist, fov_degree)
    save_path = system.get_save_dir()
    video = glob.glob(os.path.join(save_path, "*.mp4"))[0]
    return video

def launch(port):
    with gr.Blocks(
        title="TGS - Demo"
    ) as demo:
        with gr.Row(variant='panel'):
            gr.Markdown(HEADER)
    
        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                input_image = gr.Image(value=None, width=512, height=512, type="filepath", label="Input Image")
                fov_deg_slider = gr.Slider(20, 80, value=40, step=1, label="Camera Fov Degree")
                camera_dist_slider = gr.Slider(1.0, 4.0, value=1.6, step=0.1, label="Camera Distance")
                img_run_btn = gr.Button("Reconstruction")

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
                    cache_examples=False,
                    label="Examples",
                    examples_per_page=40
                )
            
            with gr.Column(scale=1):
                with gr.Row(variant='panel'):
                    seg_image = gr.Image(value=None, type="filepath", height=256, width=256, image_mode="RGBA", label="Segmented Image", interactive=False)
                    output_video = gr.Video(value=None, label="Rendered Video", height=256, autoplay=True)
                output_3dgs = Model3DGS(value=None, label="3DGS")
        
        img_run_btn.click(
            fn=preprocess,
            inputs=[input_image],
            outputs=[seg_image],
            concurrency_limit=1,
        ).success(
            fn=init_trial_dir,
            concurrency_limit=1,
        ).success(fn=run,
                inputs=[seg_image, camera_dist_slider, fov_deg_slider],
                outputs=[output_3dgs],
                concurrency_limit=1
        ).success(fn=run_video,
                inputs=[seg_image, camera_dist_slider, fov_deg_slider],
                outputs=[output_video],
                concurrency_limit=1)

        launch_args = {"server_port": port}
        demo.queue(max_size=10)
        demo.launch(auth=AUTH, **launch_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, extra = parser.parse_known_args()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch(args.port)