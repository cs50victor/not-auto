#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.3.1",
#     "torchvision==0.18.1",
#     "numpy==1.26.1",
#     "Pillow==10.3.0",
#     "huggingface_hub",
#     "safetensors",
#     "roma",
#     "gradio",
#     "matplotlib",
#     "tqdm",
#     "opencv-python",
#     "scipy",
#     "einops",
#     "trimesh",
#     "tensorboard",
#     "pyglet<2",
#     "viser==0.2.23",
#     "lpips",
#     "hydra-core",
#     "h5py",
#     "accelerate",
#     "transformers",
#     "scikit-learn",
#     "gsplat",
#     "evo",
#     "open3d",
#     "omegaconf",
#     "onnxruntime",
#     "requests",
#     "pydantic==2.10.6",
#     "gradio_client",
# ]
# ///

# https://huggingface.co/lch01/StreamVGGT
import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
import json
from sklearn.neighbors import NearestNeighbors

sys.path.append("src/")
from visual_util import predictions_to_glb
from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from streamvggt.utils.geometry import unproject_depth_map_to_point_map
from gsplat import rasterization

device = "cuda" if torch.cuda.is_available() else "mps"

print("Initializing and loading StreamVGGT model...")

local_ckpt_path = "ckpt/checkpoints.pth"
if os.path.exists(local_ckpt_path):
    print(f"Loading local checkpoint from {local_ckpt_path}")
    model = StreamVGGT()
    ckpt = torch.load(local_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    del ckpt
else:
    print("Local checkpoint not found, downloading from Hugging Face...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="lch01/StreamVGGT",
        filename="checkpoints.pth",
        revision="main",
        force_download=True
    )
    model = StreamVGGT()
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.eval() 
    del ckpt


# -------------------------------------------------------------------------
# Gaussian Extraction Functions
# -------------------------------------------------------------------------
def extract_gaussians_from_predictions(predictions, conf_threshold=0.3, use_random_quaternions=True):
    """
    Extract 3D Gaussians from StreamVGGT predictions.
    
    Args:
        predictions: Dict with 'world_points', 'world_points_conf', 'images'
        conf_threshold: Minimum confidence to include a point
        use_random_quaternions: Use random (True) or identity (False) quaternions
        
    Returns:
        List of gaussian dictionaries
    """
    pts3d = predictions["world_points"]  # [S, H, W, 3]
    confidence = predictions["world_points_conf"]  # [S, H, W]
    images = predictions["images"]  # [S, 3, H, W] or [S, H, W, 3]
    
    # Handle image format
    if images.shape[1] == 3:  # NCHW format
        images = np.transpose(images, (0, 2, 3, 1))  # Convert to NHWC
    
    gaussians = []
    
    for frame_idx in range(pts3d.shape[0]):
        frame_pts = pts3d[frame_idx]  # [H, W, 3]
        frame_conf = confidence[frame_idx]  # [H, W]
        frame_rgb = images[frame_idx]  # [H, W, 3]
        
        # Flatten spatial dimensions
        pts_flat = frame_pts.reshape(-1, 3)
        conf_flat = frame_conf.reshape(-1)
        rgb_flat = frame_rgb.reshape(-1, 3)
        
        # Filter by confidence
        valid_mask = conf_flat > conf_threshold
        valid_pts = pts_flat[valid_mask]
        valid_conf = conf_flat[valid_mask]
        valid_rgb = rgb_flat[valid_mask]
        
        if len(valid_pts) == 0:
            continue
            
        # Compute nearest neighbor distances for scale
        if len(valid_pts) > 3:
            nbrs = NearestNeighbors(n_neighbors=min(4, len(valid_pts))).fit(valid_pts)
            distances, _ = nbrs.kneighbors(valid_pts)
            if distances.shape[1] > 1:
                mean_nn_dist = distances[:, 1:].mean(axis=1)  # Skip self
            else:
                mean_nn_dist = np.ones(len(valid_pts)) * 0.002
        else:
            mean_nn_dist = np.ones(len(valid_pts)) * 0.002
        
        for i in range(len(valid_pts)):
            # Quaternion
            if use_random_quaternions:
                quat = np.random.randn(4)
                quat = quat / np.linalg.norm(quat)
            else:
                quat = np.array([1, 0, 0, 0])  # Identity
            
            # Scale based on nearest neighbors and confidence
            conf_factor = valid_conf[i]
            scale_value = mean_nn_dist[i] * (2.0 - conf_factor)
            scale = np.array([scale_value, scale_value, scale_value])
            
            # Opacity based on confidence
            opacity = 0.95 * conf_factor
            
            gaussian = {
                'position': valid_pts[i].tolist(),
                'quaternion': quat.tolist(),
                'scale': scale.tolist(),
                'opacity': float(opacity),
                'color': valid_rgb[i].tolist()
            }
            gaussians.append(gaussian)
    
    return gaussians


def save_gaussians_to_json(gaussians, output_path):
    """Save Gaussians to JSON format for external use."""
    with open(output_path, 'w') as f:
        json.dump({
            'num_gaussians': len(gaussians),
            'gaussians': gaussians
        }, f, indent=2)
    print(f"Saved {len(gaussians)} Gaussians to {output_path}")


def create_gaussian_ply(gaussians, output_path):
    """
    Create a PLY file from Gaussians for visualization.
    This is a simplified version - full 3DGS PLY has more attributes.
    """
    import struct
    
    num_gaussians = len(gaussians)
    
    # PLY header
    header = f"""ply
format binary_little_endian 1.0
element vertex {num_gaussians}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
    
    with open(output_path, 'wb') as f:
        f.write(header.encode('ascii'))
        
        for g in gaussians:
            # Position
            f.write(struct.pack('fff', *g['position']))
            # Normal (dummy)
            f.write(struct.pack('fff', 0, 0, 0))
            # Color (DC component of SH)
            f.write(struct.pack('fff', *g['color']))
            # Opacity
            f.write(struct.pack('f', g['opacity']))
            # Scale
            f.write(struct.pack('fff', *g['scale']))
            # Rotation quaternion
            f.write(struct.pack('ffff', *g['quaternion']))
    
    print(f"Saved Gaussian PLY to {output_path}")


def render_gaussian_preview(gaussians, intrinsics, width=512, height=512):
    """
    Render a preview of Gaussians using gsplat.
    Returns an RGB image.
    """
    if len(gaussians) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Convert to tensors
    positions = torch.tensor([g['position'] for g in gaussians], device=device)
    quats = torch.tensor([g['quaternion'] for g in gaussians], device=device)
    scales = torch.tensor([g['scale'] for g in gaussians], device=device)
    opacities = torch.tensor([g['opacity'] for g in gaussians], device=device)
    colors = torch.tensor([g['color'] for g in gaussians], device=device)
    
    # Simple camera at origin looking down -Z
    viewmat = torch.eye(4, device=device)[None]
    
    # Render
    try:
        rendered, _, _ = rasterization(
            positions,
            quats,
            scales,
            opacities,
            colors,
            viewmat,
            intrinsics,
            width=width,
            height=height,
            packed=False,
            render_mode="RGB"
        )
        
        # Convert to numpy image
        rendered = rendered[0].cpu().numpy()
        rendered = (rendered * 255).clip(0, 255).astype(np.uint8)
        return rendered
    except Exception as e:
        print(f"Rendering error: {e}")
        return np.zeros((height, width, 3), dtype=np.uint8)


# -------------------------------------------------------------------------
# Modified run_model to also extract Gaussians
# -------------------------------------------------------------------------
def run_model(target_dir, model) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cpu":
        raise ValueError("Neither CUDA nor MPS is available. GPU acceleration is required for this model.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    predictions = {}    
    predictions["images"] = images  # (S, 3, H, W)
    print(f"Images shape: {images.shape}")

    frames = []
    for i in range(images.shape[0]):
        image = images[i].unsqueeze(0) 
        frame = {
            "img": image
        }
        frames.append(frame)

    # Run inference
    print("Running inference...")
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                output = model.inference(frames)
    else:
        # MPS and CPU don't support autocast the same way as CUDA
        # Run without autocast for MPS/CPU
        with torch.no_grad():
            output = model.inference(frames)

    all_pts3d = []
    all_conf = []
    all_depth = []
    all_depth_conf = []
    all_camera_pose = []
    
    for res in output.ress:
        all_pts3d.append(res['pts3d_in_other_view'].squeeze(0))
        all_conf.append(res['conf'].squeeze(0))
        all_depth.append(res['depth'].squeeze(0))
        all_depth_conf.append(res['depth_conf'].squeeze(0))
        all_camera_pose.append(res['camera_pose'].squeeze(0))

    predictions["world_points"] = torch.stack(all_pts3d, dim=0)  # (S, H, W, 3)
    predictions["world_points_conf"] = torch.stack(all_conf, dim=0)  # (S, H, W)
    predictions["depth"] = torch.stack(all_depth, dim=0)  # (S, H, W, 1)
    predictions["depth_conf"] = torch.stack(all_depth_conf, dim=0)  # (S, H, W)
    predictions["pose_enc"] = torch.stack(all_camera_pose, dim=0)  # (S, 9)

    print("World points shape:", predictions["world_points"].shape)
    print("World points confidence shape:", predictions["world_points_conf"].shape)
    print("Depth map shape:", predictions["depth"].shape)
    print("Depth confidence shape:", predictions["depth_conf"].shape)
    print("Pose encoding shape:", predictions["pose_enc"].shape)
    print(f"Images shape: {images.shape}")
    
    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"].unsqueeze(0) if predictions["pose_enc"].ndim == 2 else predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic.squeeze(0)  # (S, 3, 4)
    predictions["intrinsic"] = intrinsic.squeeze(0) if intrinsic is not None else None  # (S, 3, 3) or None
    print("Extrinsic shape:", predictions["extrinsic"].shape)
    print("Intrinsic shape:", predictions["intrinsic"].shape)

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy()

    predictions["world_points_from_depth"] = predictions["world_points"]

    # Clean up
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    return predictions


def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
    output_gaussians=True,
):
    """
    Perform reconstruction using the already-created target_dir/images.
    Now also outputs Gaussians if requested.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None, None

    start_time = time.time()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Extract and save Gaussians
    gaussian_info = None
    if output_gaussians:
        print("Extracting Gaussians from predictions...")
        gaussians = extract_gaussians_from_predictions(
            predictions, 
            conf_threshold=(100 - conf_thres) / 100.0  # Convert from percentage
        )
        
        gaussian_json_path = os.path.join(target_dir, "gaussians.json")
        save_gaussians_to_json(gaussians, gaussian_json_path)
        
        gaussian_ply_path = os.path.join(target_dir, "gaussians.ply")
        create_gaussian_ply(gaussians, gaussian_ply_path)
        
        if predictions["intrinsic"] is not None:
            intrinsics = torch.tensor(predictions["intrinsic"][0:1], device=device)
            preview = render_gaussian_preview(gaussians, intrinsics)
            preview_path = os.path.join(target_dir, "gaussian_preview.png")
            cv2.imwrite(preview_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        
        gaussian_info = f"Generated {len(gaussians)} Gaussians\nFiles saved: gaussians.json, gaussians.ply"

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    del predictions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."
    if gaussian_info:
        log_msg += f"\n{gaussian_info}"

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True), gaussian_info


# -------------------------------------------------------------------------
# Helper functions (unchanged from original)
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images):
    """Create a new 'target_dir' + 'images' subfolder"""
    start_time = time.time()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    image_paths = sorted(image_paths)
    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


def update_gallery_on_upload(input_video, input_images):
    """Update gallery when files are uploaded."""
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


def clear_fields():
    """Clear the 3D viewer and gallery."""
    return None


def update_log():
    """Display loading message."""
    return "Loading and Reconstructing..."


# -------------------------------------------------------------------------
# Build Enhanced Gradio UI with Gaussian output
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(
    theme=theme,
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    
    .gaussian-info * {
        font-family: monospace;
        font-size: 14px !important;
        color: #10b981 !important;
        background-color: #f0fdf4;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #86efac;
    }
    """,
) as demo:
    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")

    gr.HTML(
        """
    <h1>StreamVGGT with Gaussian Splatting Output</h1>
    <p>
    <a href="https://github.com/wzzheng/StreamVGGT">GitHub Repository</a> |
    <a href="https://wzzheng.net/StreamVGGT/">Project Page</a> |
    <a href="https://arxiv.org/abs/2507.11539">Paper</a>
    </p>

    <div style="font-size: 16px; line-height: 1.5;">
    <p><strong>Enhanced with 3D Gaussian Splatting output!</strong> This demo now extracts 3D Gaussians from StreamVGGT predictions, 
    enabling seamless integration with modern rendering pipelines and scene expansion workflows.</p>
    
    <h3>What's New:</h3>
    <ul>
        <li> -  <strong>Gaussian Extraction:</strong> Automatically converts point clouds to 3D Gaussians</li>
        <li> -  <strong>Multiple Output Formats:</strong> JSON for data exchange, PLY for visualization</li>
        <li> -  <strong>Smart Initialization:</strong> Uses nearest-neighbor scaling and confidence weighting</li>
        <li> -  <strong>Ready for Scene Expansion:</strong> Optimized for boundary blending in streaming scenarios</li>
    </ul>
    </div>
    """
    )

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)

            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )

        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (Point Cloud and Gaussians)**")
                log_output = gr.Markdown(
                    "Please upload a video or images, then click Reconstruct.", elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)
                
                # New: Gaussian info display
                gaussian_info_output = gr.Markdown(
                    "", 
                    elem_classes=["gaussian-info"],
                    visible=False
                )

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery, gaussian_info_output],
                    scale=1,
                )

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Pointmap Branch"],
                    label="Select a Prediction Mode",
                    value="Depthmap and Camera Branch",
                    scale=1,
                )
                output_gaussians = gr.Checkbox(
                    label="Generate Gaussian Outputs",
                    value=True,
                    info="Export 3D Gaussians for rendering"
                )

            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                with gr.Column():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

    # Download section for Gaussian files
    with gr.Row():
        gr.Markdown("### Download Gaussian Files")
        download_json = gr.File(label="Gaussian JSON", visible=False)
        download_ply = gr.File(label="Gaussian PLY", visible=False)

    # Reconstruct button with Gaussian output
    def gradio_demo_with_gaussian(
        target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, 
        show_cam, mask_sky, prediction_mode, output_gaussians
    ):
        result = gradio_demo(
            target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg,
            show_cam, mask_sky, prediction_mode, output_gaussians
        )
        
        if len(result) == 4:
            glbfile, log_msg, dropdown, gaussian_info = result
            
            # Make Gaussian files available for download
            download_files = {}
            if gaussian_info and output_gaussians:
                json_path = os.path.join(target_dir, "gaussians.json")
                ply_path = os.path.join(target_dir, "gaussians.ply")
                
                download_files['json'] = json_path if os.path.exists(json_path) else None
                download_files['ply'] = ply_path if os.path.exists(ply_path) else None
                
                return (
                    glbfile, log_msg, dropdown, 
                    gr.Markdown(gaussian_info, visible=True),
                    gr.File(value=download_files['json'], visible=True),
                    gr.File(value=download_files['ply'], visible=True)
                )
            
            return glbfile, log_msg, dropdown, gr.Markdown("", visible=False), None, None
        else:
            return result[0], result[1], result[2], gr.Markdown("", visible=False), None, None

    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo_with_gaussian,
        inputs=[
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            output_gaussians,
        ],
        outputs=[reconstruction_output, log_output, frame_filter, gaussian_info_output, download_json, download_ply],
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]
    )

    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(show_error=True, share=True)