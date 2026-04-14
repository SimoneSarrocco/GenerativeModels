# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -

#
# # Diffusion Models for Implicit Image Segmentation Ensembles<br>
# <br>
# This tutorial illustrates how to use MONAI for 2D segmentation of images using DDPMs, as proposed in [1].<br>
# The same structure can also be used for conditional image generation, or image-to-image translation, as proposed in [2,3].
# <br>
# <br>
# [1] - Wolleb et al. "Diffusion Models for Implicit Image Segmentation Ensembles", https://arxiv.org/abs/2112.03145<br>
# [2] - Waibel et al. "A Diffusion Model Predicts 3D Shapes from 2D Microscopy Images", https://arxiv.org/abs/2208.14125<br>
# [3] - Durrer et al. "Diffusion Models for Contrast Harmonization of Magnetic Resonance Images", https://aps.arxiv.org/abs/2303.08189
#
#

# ## Setup environment

# !python -c "import monai" || pip install -q "monai-weekly[pillow, tqdm]"
# !python -c "import matplotlib" || pip install -q matplotlib
# !python -c "import seaborn" || pip install -q seaborn

#
# ## Setup imports

# +
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tempfile
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import set_determinism
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from dataset import OCTDataset
from new_dataset import OCTFolderDataset
from PIL import Image
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddpm import DDPMScheduler
from generative.losses import PerceptualLoss
from generative.metrics.ssim import SSIMMetric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import cv2
import torch.distributed as dist
import pandas as pd
from pathlib import Path


def _plot_to_image(fig):
    fig.canvas.draw()
    # Use buffer_rgba instead of tostring_rgb
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Include alpha channel
    return data[..., :3].transpose(2, 0, 1)  # Exclude alpha channel and reorder dimensions


def psnr(output: torch.Tensor, target: torch.Tensor) -> float:
    assert target.shape == output.shape
    mse = mean_flat((target - output) ** 2)
    target_image = np.asarray(target.cpu(), dtype=np.float32)
    psnr = 20 * math.log(np.max(target_image), 10) - 10 * math.log(mse.cpu().mean(), 10)
    return psnr


def save_diff_map_bwr_fixed(generated: np.ndarray,
                            art100_ref: np.ndarray,
                            save_path: str | Path):
    """
    Save (generated - ART100) map with fixed range [-1, 1].
    Both inputs can be 0..1 or 0..255; automatically aligned.
    """
    gen = np.asarray(generated, dtype=np.float32)
    ref = np.asarray(art100_ref, dtype=np.float32)

    # Range normalization: align both to [0, 1]
    if gen.max() > 1.1:
        gen = gen / 255.0
    if ref.max() > 1.1:
        ref = ref / 255.0

    diff = gen - ref

    # Fixed symmetric range for comparability
    vmin, vmax = -1.0, 1.0

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(diff, cmap='bwr', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Generated - ART100 (fixed [-1,1])')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def save_difference_map_inference(generated, art100_ref, save_dir, step, vmin=-1, vmax=1, cmap='bwr'):
    """
    Save a difference map between generated output and ART100 reference.
    Both arrays should be float32 in [0,1] or [0,255].
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert to float32 NumPy arrays
    gen = np.asarray(generated, dtype=np.float32)
    ref = np.asarray(art100_ref, dtype=np.float32)

    # If one is 0-1 and the other 0-255, rescale for consistency
    if gen.max() <= 1.1 and ref.max() > 1.1:
        gen = gen * 255.0
    elif ref.max() <= 1.1 and gen.max() > 1.1:
        ref = ref * 255.0

    # Compute difference
    diff = gen - ref

    # Clip range symmetrically
    vmax = vmax or np.max(np.abs(diff))
    vmin = -vmax

    # Plot and save
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(diff, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Generated - ART100 (intensity)')
    plt.title(f'Difference Map - Output vs ART100 (B-scan {step+1})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'diff_bscan_{step+1:03d}.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)


def load_art100_refs(art100_root_dir: str | Path) -> dict[str, np.ndarray]:
    """
    Load one ART100 reference per location folder (assumes first .tiff is the ref).
    Returns dict like {'0000': img, '0001': img, ..., '0020': img}, float32 0..255.
    """
    art100_root = Path(art100_root_dir)
    refs = {}
    for loc_dir in sorted(art100_root.iterdir()):
        if not loc_dir.is_dir():
            continue
        tiffs = sorted(loc_dir.glob("*.tiff"))
        if not tiffs:
            print(f"⚠️ No ART100 TIFF found in {loc_dir}")
            continue
        refs[loc_dir.name] = np.array(Image.open(tiffs[0]).convert("L"), dtype=np.float32)
    if len(refs) != 21:
        print(f"⚠️ Loaded {len(refs)} ART100 refs (expected 21).")
    return refs

def save_diff_map_bwr(generated: np.ndarray,
                      art100_ref: np.ndarray,
                      save_path: str | Path,
                      vspan: float | None = None):
    """
    Save a (generated - ART100) map with 'bwr' colormap.
    Both arrays can be 0..1 or 0..255; this function auto-aligns ranges.
    """
    gen = np.asarray(generated, dtype=np.float32)
    ref = np.asarray(art100_ref, dtype=np.float32)

    # Auto range alignment (either both 0..1 or both 0..255)
    if gen.max() <= 1.1 and ref.max() > 1.1:
        ref = ref / 255.0
    elif ref.max() <= 1.1 and gen.max() > 1.1:
        gen = gen * 255.0

    diff = gen - ref

    # Symmetric color scaling
    if vspan is None:
        vmax = np.max(np.abs(diff))
        if vmax < 1e-6:
            vmax = 1.0
    else:
        vmax = float(vspan)
    vmin = -vmax

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(diff, cmap='bwr', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Generated - ART100')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def resume_training(model, optimizer, scheduler, checkpoint_path, start_epoch=0):
    """
    Resume training from a checkpoint.

    Args:
        model: The DiffusionModelUNet model
        optimizer: The optimizer (Adam)
        scheduler: The DDPMScheduler
        checkpoint_path: Path to the checkpoint file
        start_epoch: The epoch to resume from (default: 0, determined from checkpoint filename if possible)

    Returns:
        model: The loaded model
        optimizer: The loaded optimizer if optimizer state was saved
        start_epoch: The epoch to resume from
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    # Extract epoch number from filename if not provided
    if start_epoch == 0 and "epoch" not in checkpoint_path:
        try:
            # Try to extract epoch number from filename (e.g., ddpm_oct_model_50.pt → 50)
            filename = os.path.basename(checkpoint_path)
            epoch_str = filename.split('_')[-1].split('.')[0]
            if epoch_str.isdigit():
                start_epoch = int(epoch_str)
            print(f"Extracted start epoch: {start_epoch}")
        except:
            print("Could not extract epoch number from filename. Starting from provided start_epoch.")

    # Load the state dict
    checkpoint = torch.load(checkpoint_path)

    # If checkpoint is just the model state dict
    if isinstance(checkpoint, dict) and all(k.startswith("module.") or "." in k for k in checkpoint.keys()):
        model.load_state_dict(checkpoint)
        print("Loaded model weights only.")
        return model, optimizer, start_epoch

    # If checkpoint contains more info (full training state)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1  # +1 because we want to start from the next epoch
        if "scheduler_state" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        print(f"Loaded complete training state. Resuming from epoch {start_epoch}")
        return model, optimizer, start_epoch

    print("Loaded model weights. Optimizer and scheduler states not found.")
    return model, optimizer, start_epoch


def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir, is_final=False):
    """
    Save a checkpoint with full training state.

    Args:
        model: The model to save
        optimizer: The optimizer to save
        scheduler: The noise scheduler
        epoch: Current epoch number
        checkpoint_dir: Directory to save the checkpoint
        is_final: Whether this is the final checkpoint of training
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state": scheduler.get_state() if hasattr(scheduler, "get_state") else None,
    }

    if is_final:
        checkpoint_path = f"{checkpoint_dir}/ddpm_oct_model_final.pt"
    else:
        checkpoint_path = f"{checkpoint_dir}/ddpm_oct_model_epoch_{epoch}.pt"

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")


torch.multiprocessing.set_sharing_strategy("file_system")
print_config()
# -

# ## Setup data directory

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory


#
# ## Set deterministic training for reproducibility

set_determinism(1927)

#
# # Preprocessing of the BRATS Dataset in 2D slices for training
# We download the BRATS training dataset from the Decathlon dataset. \
# We slice the volumes in axial 2D slices, and assign slice-wise ground truth segmentations of the tumor to all slices.
# Here we use transforms to augment the training dataset:
#
# 1. `LoadImaged` loads the brain MR images from files.
# 1. `EnsureChannelFirstd` ensures the original data to construct "channel first" shape.
# 1. `ScaleIntensityRangePercentilesd` takes the lower and upper intensity percentiles and scales them to [0, 1].
#

# +
"""train = np.load('/home/simone.sarrocco/thesis/project/data/train_set_patient_split.npz')['images']
val = np.load('/home/simone.sarrocco/thesis/project/data/val_set_patient_split.npz')['images']
test = np.load('/home/simone.sarrocco/thesis/project/data/test_set_patient_split.npz')['images']"""


# Path to your ART10_21Bscans folder
art_folder = "/home/simone.sarrocco/thesis/project/data/art_1_10_100/Nifty/Images_by_location/ART10_21Bscans"

# Transform: convert to tensor and scale to [0, 1]
transform = transforms.Compose([transforms.ToTensor()])

# Create dataset
test_data = OCTFolderDataset(root_folder=art_folder, num_inputs=10, padding=(0,8,0,8))

# Create DataLoader
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

# train_data_split = torch.tensor(train).view((-1, 1, 496, 768))  # Pass shape as a tuple
# val_data_split = torch.tensor(val).view((-1, 1, 496, 768))  # Pass shape as a tuple
# test_data_split = torch.tensor(test).view((-1, 1, 496, 768))  # Pass shape as a tuple

# final_val_data_split = torch.cat([val_data_split, test_data_split], dim=0)

"""# train_data = OCTDataset(train_data_split, transform=True)
train_data = OCTDataset(train, transform=True, pixel_range=0)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
# print(f'Shape of training set: {train_data_split.shape}')
print(f'Shape of training set: {train.shape}')

# val_data = OCTDataset(final_val_data_split)
val_data = OCTDataset(val, transform=False, pixel_range=0)
# val_datalist = [{"image": val[i, -1:, ...]} for i in range(len(val))]
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
# print(f'Shape of validation set: {val_data_split.shape}')
print(f'Shape of validation set: {val.shape}')"""

# test_data = OCTDataset(test, transform=False, pixel_range=0)
# val_datalist = [{"image": val[i, -1:, ...]} for i in range(len(val))]
# test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
# print(f'Shape of validation set: {val_data_split.shape}')
# print(f'Shape of testing set: {test_data.shape}')
print(f"Total samples: {len(test_data)}")  # Expect len = (#locations × 10)


run_number = '7th_run'
#
# ## Define network, scheduler, optimizer, and inferer
#
# At this step, we instantiate the MONAI components to create a DDPM, the UNET, the noise scheduler, and the inferer used for training and sampling. We are using the DDPM scheduler containing 1000 timesteps, and a 2D UNET with attention mechanisms in the 3rd level (`num_head_channels=64`).<br>
#
writer = SummaryWriter(log_dir=f'/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/image_to_image_translation/logs/{run_number}/new_acquired_data/art10_inference_again')
device = torch.device("cuda")

checkpoint_dir = f"/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/image_to_image_translation/checkpoints/{run_number}"
os.makedirs(checkpoint_dir, exist_ok=True)

output_dir = f"/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/image_to_image_translation/outputs/{run_number}/new_acquired_data/art10_inference_again"
os.makedirs(output_dir, exist_ok=True)

PSNR = PeakSignalNoiseRatio(data_range=1.).to(device)
# SSIM = StructuralSimilarityIndexMeasure().to(device)
SSIM = SSIMMetric(spatial_dims=2, data_range=1.)
LPIPS = PerceptualLoss(spatial_dims=2, device=device, network_type='resnet50', pretrained=True, pretrained_path='/home/simone.sarrocco/thesis/project/models/lpips_training/checkpoints/my_resnet_7.pth')
LPIPS_RAD = PerceptualLoss(spatial_dims=2, device=device, network_type='radimagenet_resnet50')
LPIPS_RESNET = PerceptualLoss(spatial_dims=2, device=device, network_type='resnet50')
# LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

"""
resume_training_flag = False  # Set to True to resume training, False to start from scratch
checkpoint_path = "/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/image_to_image_translation/checkpoints/4th_run/ddpm_oct_model_601.pt"

model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=2,
    out_channels=1,
    num_channels=(64, 128, 128, 256, 256, 512, 512),
    attention_levels=(False, False, False, False, False, False, True),
    num_res_blocks=2,
    num_head_channels=1,
    with_conditioning=False,
)
model.to(device)

scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="v_prediction", beta_start=0.00085, beta_end=0.0120)
optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-5)
inferer = DiffusionInferer(scheduler)

#
# ### Model training of the Diffusion Model<br>
# We train our diffusion model for 4000 epochs.\
# In every step, we concatenate the original MR image to the noisy segmentation mask, to predict a slightly denoised segmentation mask.\
# This is described in Equation 7 of the paper https://arxiv.org/pdf/2112.03145.pdf.

n_epochs = 1000
val_interval = 25
epoch_loss_list = []
val_epoch_loss_list = []
val_sample = 100
save_interval = 100

# +
validation_samples_path = f'/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/image_to_image_translation/results/{run_number}/validation/output_samples'
os.makedirs(validation_samples_path, exist_ok=True)

scaler = GradScaler('cuda')
total_start = time.time()
i = 0

epoch_psnr, epoch_ssim, epoch_my_psnr = [], [], []

# Resume from checkpoint if flag is set
if resume_training_flag and os.path.exists(checkpoint_path):
    model, optimizer, start_epoch = resume_training(model, optimizer, scheduler, checkpoint_path)

    # Load the loss history if available (optional)
    loss_history_path = os.path.join(os.path.dirname(checkpoint_path), "loss_history.npz")
    if os.path.exists(loss_history_path):
        history = np.load(loss_history_path)
        epoch_loss_list = history["train_loss"].tolist()
        if "val_loss" in history:
            val_epoch_loss_list = history["val_loss"].tolist()
        print(f"Loaded loss history with {len(epoch_loss_list)} entries")

    # Calculate global step for tensorboard
    i = start_epoch * len(train_loader)
    print(f"Resuming training from epoch {start_epoch} (global step {i})")
else:
    print("Starting training from scratch")

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, (art10, pseudoart100) in progress_bar:
        art10 = art10.to(device)
        pseudoart100 = pseudoart100.to(device)
        # seg = data["label"].to(device)  # this is the ground truth segmentation
        optimizer.zero_grad(set_to_none=True)
        # timesteps = torch.randint(0, 1000, (art10.shape[0],)).to(device)  # pick a random time step t
        # timesteps = torch.randint(
        #    0, inferer.scheduler.num_train_timesteps, (art10.shape[0],), device=art10.device
        # ).long()


        with autocast('cuda', enabled=True):
            # Generate random noise
            # noise = torch.randn_like(pseudoart100).to(device)
            # noisy_pseudoart100 = scheduler.add_noise(
            #    original_samples=pseudoart100, noise=noise, timesteps=timesteps
            # )  # we only add noise to the segmentation mask
            noise = torch.randn_like(pseudoart100).to(device)
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (pseudoart100.shape[0],), device=pseudoart100.device
            ).long()

            # Get target for the v-prediction parameterization
            target = inferer.scheduler.get_velocity(pseudoart100, noise, timesteps)

            # combined = torch.cat(
            #    (art10, target), dim=1
            # )  # we concatenate the brain MR image with the noisy segmenatation mask, to condition the generation process

            # Get model prediction
            noise_pred = inferer(inputs=pseudoart100, condition=art10, mode="concat", diffusion_model=model, noise=noise, timesteps=timesteps)

            # prediction = model(x=combined, timesteps=timesteps)
            # Get model prediction
            loss = F.mse_loss(noise_pred.float(), target.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        i += 1
        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        writer.add_scalar('Loss/train', loss.item(), i)

    epoch_loss_list.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_epoch_loss = 0
        for step, (art10, pseudoart100) in enumerate(val_loader):
            art10 = art10.to(device)
            pseudoart100 = pseudoart100.to(device)
            with torch.no_grad():
                with autocast(enabled=True, device_type='cuda'):
                    noise = torch.randn_like(pseudoart100).to(device)
                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps, (pseudoart100.shape[0],), device=pseudoart100.device
                    ).long()
                    target = inferer.scheduler.get_velocity(pseudoart100, noise, timesteps)
                    # combined = torch.cat(
                    #    (art10, target), dim=1
                    # )
                    noise_pred = inferer(inputs=pseudoart100, diffusion_model=model, noise=noise, timesteps=timesteps, condition=art10, mode="concat")
                    val_loss = F.mse_loss(noise_pred.float(), target.float())

            val_epoch_loss += val_loss.item()
            progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
        val_epoch_loss_list.append(val_epoch_loss / (step + 1))

        # Sampling image during training
        noise = torch.randn_like(pseudoart100)
        noise = noise.to(device)
        scheduler.set_timesteps(num_inference_steps=1000)
        # combined_noise = torch.cat(
        #    (art10, noise), dim=1
        #)
        with autocast(enabled=True, device_type='cuda'):
            image = inferer.sample(input_noise=noise, conditioning=art10, mode="concat", diffusion_model=model, scheduler=scheduler)
            # print(f'Pixel range of output image from training set: {torch.min(image)}, {torch.max(image)}')
            art10_to_plot = art10.squeeze(0).cpu()
            pseudoart100_to_plot = pseudoart100.squeeze(0).cpu()
            image_to_plot = image.squeeze(0).cpu()
            writer.add_image(f'Training/Input', art10_to_plot, epoch + 1)
            writer.add_image(f'Training/Sample', image_to_plot, epoch + 1)
            writer.add_image(f'Training/Target', pseudoart100_to_plot, epoch + 1)

        # plt.figure(figsize=(2, 2))
        # plt.imshow(image[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
        # plt.tight_layout()
        # plt.axis("off")
        # plt.show()

    if (epoch+1) % val_sample == 0:
        model.eval()
        mse_batches, psnr_batches, ssim_batches, my_psnr_batches = [], [], [], []
        for step, (art10, pseudoart100) in enumerate(val_loader):
            art10 = art10.to(device)
            pseudoart100 = pseudoart100.to(device)
            # timesteps = torch.randint(0, 1000, (art10.shape[0],)).to(device)
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (pseudoart100.shape[0],), device=pseudoart100.device
            ).long()
            noise = torch.randn_like(pseudoart100).to(device)
            # current_img = noise  # for the pseudoART100, we start from random noise.
            # combined = torch.cat(
            #    (art10, noise), dim=1
            # )  # We concatenate the input ART10 to add anatomical information.

            scheduler.set_timesteps(num_inference_steps=1000)
            # progress_bar = tqdm(scheduler.timesteps)
            # chain = torch.zeros(current_img.shape)

            with autocast(enabled=True, device_type='cuda'):
                with torch.no_grad():
                    current_img = inferer.sample(input_noise=noise, conditioning=art10, mode="concat", diffusion_model=model,
                                                 scheduler=scheduler)
                    # print(f'Pixel range of output_image {step+1} from validation_set: {torch.min(current_img)}, {torch.max(current_img)}')
                    # Linearly scale from [-1,1] to [0,1]
                    # art10_scaled = (art10 + 1) / 2
                    # pseudoart100_scaled = (pseudoart100 + 1) / 2
                    # current_img_scaled = (current_img + 1) / 2
                    art10_scaled = art10
                    pseudoart100_scaled = pseudoart100
                    current_img_scaled = current_img

                    if (step + 1) % 5 == 0:
                        images_stacked = torch.stack(
                            [art10_scaled[0, :, 8:-8, :], pseudoart100_scaled[0, :, 8:-8, :],
                             current_img_scaled[0, :, 8:-8, :]], dim=0)
                        grid = make_grid(images_stacked, nrow=3, normalize=False, value_range=None, padding=0)
                        writer.add_image(f'Validation/Sample_{step + 1}', grid, epoch + 1)

                    # Compute PSNR, SSIM, MSE, and LPIPS between target image (pseudoART100) and denoised image (current_img)
                    mse_batch = mean_flat((current_img[:, :, 8:-8, :] - pseudoart100[:, :, 8:-8, :]) ** 2)
                    psnr_batch = PSNR(current_img[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])
                    my_psnr_batch = psnr(current_img[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])
                    ssim_batch = SSIM(current_img[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])

                    mse_batches.append(mse_batch.mean().cpu())
                    psnr_batches.append(psnr_batch.cpu())
                    my_psnr_batches.append(my_psnr_batch)
                    ssim_batches.append(ssim_batch.cpu())

        psnr_batches = np.asarray(psnr_batches, dtype=np.float32)
        my_psnr_batches = np.asarray(my_psnr_batches, dtype=np.float32)
        ssim_batches = np.asarray(ssim_batches, dtype=np.float32)
        mse_batches = np.asarray(mse_batches, dtype=np.float32)

        # Calculate averages
        avg_psnr, std_psnr = np.mean(psnr_batches), np.std(psnr_batches)
        avg_my_psnr, std_my_psnr = np.mean(my_psnr_batches), np.std(my_psnr_batches)
        avg_ssim, std_ssim = np.mean(ssim_batches), np.std(ssim_batches)
        avg_mse, std_mse = np.mean(mse_batches), np.std(mse_batches)

        # Append average PSNR and SSIM for the current epoch
        epoch_psnr.append(avg_psnr)
        epoch_my_psnr.append(avg_my_psnr)
        epoch_ssim.append(avg_ssim)

        # Log average metrics to TensorBoard
        metrics_summary = {
            "PSNR": avg_psnr,
            "SSIM": avg_ssim,
            "MSE": avg_mse,
            # "PERC_LOSS": avg_perceptual,
        }

        for metric_name, value in metrics_summary.items():
            writer.add_scalar(f"Validation_metrics/{metric_name}", value.item(), epoch + 1)
        print(
            f"Validation metrics, epoch {epoch + 1}: PSNR: {avg_psnr.item():.4f} ± {std_psnr.item():.4f} | MY_PSNR: {avg_my_psnr.item():.4f} ± {std_my_psnr.item():.4f}| SSIM: {avg_ssim.item():.4f} ± {std_ssim.item():.4f} | MSE: {avg_mse.item():.4f} ± {std_mse.item():.4f}")
        if epoch_psnr[-1] >= np.max(np.asarray(epoch_psnr, dtype=np.float32)) or epoch_ssim[-1] >= np.max(np.asarray(epoch_ssim, dtype=np.float32)):
            # save checkpoint if either PSNR or SSIM improved from last validation
            save_checkpoint(model, optimizer, scheduler, epoch+1, checkpoint_dir)

            # Optionally save loss history
            np.savez(
                f"{checkpoint_dir}/loss_history.npz",
                train_loss=np.array(epoch_loss_list),
                val_loss=np.array(val_epoch_loss_list) if val_epoch_loss_list else np.array([]),
            )

    # Save checkpoint at regular intervals
    # if (epoch + 1) % save_interval == 0:
    #    save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir)

        # Optionally save loss history
    #    np.savez(
    #        f"{checkpoint_dir}/loss_history.npz",
    #        train_loss=np.array(epoch_loss_list),
    #        val_loss=np.array(val_epoch_loss_list) if val_epoch_loss_list else np.array([]),
    #    )

# torch.save(model.state_dict(), f"{checkpoint_dir}/ddpm_oct_model_last_epoch.pt")
save_checkpoint(model, optimizer, scheduler, n_epochs - 1, checkpoint_dir, is_final=True)
total_time = time.time() - total_start
print(f"train diffusion completed, total time: {total_time}.")
#plt.style.use("seaborn-bright")
#plt.title("Learning Curves Diffusion Model", fontsize=20)
#plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_loss_list, color="C0", linewidth=2.0, label="Train")
#plt.plot(
#    np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
#    val_epoch_loss_list,
#    color="C1",
#    linewidth=2.0,
#    label="Validation",
#)
#plt.yticks(fontsize=12)
#plt.xticks(fontsize=12)
#plt.xlabel("Epochs", fontsize=16)
#plt.ylabel("Loss", fontsize=16)
#plt.legend(prop={"size": 14})
#plt.savefig('/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/image_to_image_translation/results/losses.png')
#plt.close()
# plt.show()
# -

#
# # Sampling of a new segmentation mask for an input image of the validation set<br>
#
# Starting from random noise, we want to generate a segmentation mask for a brain MR image of our validation set.\
# Due to the stochastic generation process, we can sample an ensemble of n different segmentation masks per MR image.\
# First, we pick an image of our validation set, and check the ground truth segmentation mask.


# +
idx = 0
data = val_data[idx]
inputimg = data["image"][0, ...]  # Pick an input slice of the validation set to be segmented
inputlabel = data["label"][0, ...]  # Check out the ground truth label mask. If it is empty, pick another input slice.


plt.figure("input" + str(inputlabel))
plt.imshow(inputimg, vmin=0, vmax=1, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure("input" + str(inputlabel))
plt.imshow(inputlabel, vmin=0, vmax=1, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()


model.eval()


# -

# Then we set the number of samples in the ensemble n. \
# Starting from the input image (which ist the brain MR image), we follow Algorithm 1 of the paper "Diffusion Models for Implicit Image Segmentation Ensembles" (https://arxiv.org/pdf/2112.03145.pdf) n times.\
# This gives us an ensemble of n different predicted segmentation masks.

n = 5
input_img = inputimg[None, None, ...].to(device)
ensemble = []
for k in range(5):
    noise = torch.randn_like(input_img).to(device)
    current_img = noise  # for the segmentation mask, we start from random noise.
    combined = torch.cat(
        (input_img, noise), dim=1
    )  # We concatenate the input brain MR image to add anatomical information.

    scheduler.set_timesteps(num_inference_steps=1000)
    progress_bar = tqdm(scheduler.timesteps)
    chain = torch.zeros(current_img.shape)
    for t in progress_bar:  # go through the noising process
        with autocast(enabled=False):
            with torch.no_grad():
                model_output = model(combined, timesteps=torch.Tensor((t,)).to(current_img.device))
                current_img, _ = scheduler.step(
                    model_output, t, current_img
                )  # this is the prediction x_t at the time step t
                if t % 100 == 0:
                    chain = torch.cat((chain, current_img.cpu()), dim=-1)
                combined = torch.cat(
                    (input_img, current_img), dim=1
                )  # in every step during the denoising process, the brain MR image is concatenated to add anatomical information

    plt.style.use("default")
    plt.imshow(chain[0, 0, ..., 64:].cpu(), vmin=0, vmax=1, cmap="gray")
    plt.tight_layout()
    plt.axis("off")
    plt.show()
    ensemble.append(current_img)  # this is the output of the diffusion model after T=1000 denoising steps


#
# ## Segmentation prediction
# The predicted segmentation mask is obtained from the output of the diffusion model by thresholding.\
# We compute the Dice score for all predicted segmentations of the ensemble, as well as the pixel-wise mean and the variance map over the ensemble.\
# As shown in the paper "Diffusion Models for Implicit Image Segmentation Ensembles" (https://arxiv.org/abs/2112.03145), we see that taking the mean over n=5 samples improves the segmentation performance.\
# The variance maps highlights pixels where the model is unsure about it's own prediction.
#
#


def dice_coeff(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2.0 * intersection.sum() / im_sum


# +
for i in range(len(ensemble)):
    prediction = torch.where(ensemble[i] > 0.5, 1, 0).float()  # a binary mask is obtained via thresholding
    score = dice_coeff(
        prediction[0, 0].cpu(), inputlabel.cpu()
    )  # we compute the dice scores for all samples separately
    print("Dice score of sample" + str(i), score)


E = torch.where(torch.cat(ensemble) > 0.5, 1, 0).float()
var = torch.var(E, dim=0)  # pixel-wise variance map over the ensemble
mean = torch.mean(E, dim=0)  # pixel-wise mean map over the ensemble
mean_prediction = torch.where(mean > 0.5, 1, 0).float()

score = dice_coeff(mean_prediction[0, ...].cpu(), inputlabel.cpu())  # Here we predict the Dice score for the mean map
print("Dice score on the mean map", score)

plt.style.use("default")
plt.imshow(mean[0, ...].cpu(), vmin=0, vmax=1, cmap="gray")  # We plot the mean map
plt.tight_layout()
plt.axis("off")
plt.show()
plt.style.use("default")
plt.imshow(var[0, ...].cpu(), vmin=0, vmax=1, cmap="jet")  # We plot the variance map
plt.tight_layout()
plt.axis("off")
plt.show()
"""


model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=2,
    out_channels=1,
    num_channels=(64, 128, 128, 256, 256, 512, 512),
    attention_levels=(False, False, False, False, False, False, True),
    num_res_blocks=2,
    num_head_channels=1,
    with_conditioning=False,
)
model.to(device)

scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="v_prediction", beta_start=0.00085, beta_end=0.0120)
optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-5)
inferer = DiffusionInferer(scheduler)
scaler = GradScaler('cuda')

# Load weights
ckpt_path = "/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/image_to_image_translation/checkpoints/7th_run/ddpm_oct_model_epoch_300.pt"
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
os.makedirs('/home/simone.sarrocco/thesis/project/visual_turing_test/images/DDPM_7th_new/new_acquired_ART10_again', exist_ok=True)
# os.makedirs('/home/simone.sarrocco/thesis/project/visual_turing_test/images/DDPM_7th_TIFF', exist_ok=True)

epoch = 299
all_images = []
mse_batches, psnr_batches, ssim_batches, my_psnr_batches = [], [], [], []
lpips_batches, lpips_rad_batches, pseudo_lpips_batches = [], [], []

mse_batches_valid_region, ssim_batches_valid_region, psnr_batches_valid_region, my_psnr_batches_valid_region = [], [], [], []
lpips_batches_valid_region, lpips_rad_batches_valid_region, pseudo_lpips_batches_valid_region = [], [], []

mse_batches_masking_input_artifacts, psnr_batches_masking_input_artifacts, my_psnr_batches_masking_input_artifacts, ssim_batches_masking_input_artifacts = [], [], [], []
lpips_batches_masking_input_artifacts, lpips_rad_batches_masking_input_artifacts, pseudo_lpips_batches_masking_input_artifacts = [], [], []

count = 2
folder = 1307
index = 0

# ==============================
# Load ART100 reference B-scans (one per location)
# ==============================
# One ref per location (0000..0020)
art100_refs = load_art100_refs(
    "/home/simone.sarrocco/thesis/project/data/art_1_10_100/Nifty/Images_by_location/ART100_21Bscans"
)

"""# Create a list to store per-image metric results
metrics_per_image = []
for step, (art10, pseudoart100, _) in enumerate(test_loader):
    if step % 10 == 0 and step != 0:
        count += 1
        folder += 1
        index = 0
    art10 = art10.to(device)
    pseudoart100 = pseudoart100.to(device)
    # timesteps = torch.randint(0, 1000, (art10.shape[0],)).to(device)
    timesteps = torch.randint(
        0, inferer.scheduler.num_train_timesteps, (pseudoart100.shape[0],), device=pseudoart100.device
    ).long()
    noise = torch.randn_like(pseudoart100).to(device)
    # current_img = noise  # for the pseudoART100, we start from random noise.
    # combined = torch.cat(
    #    (art10, noise), dim=1
    # )  # We concatenate the input ART10 to add anatomical information.

    scheduler.set_timesteps(num_inference_steps=1000)
    progress_bar = tqdm(scheduler.timesteps)
    # chain = torch.zeros(current_img.shape)

    with autocast(enabled=True, device_type='cuda'):
        with torch.no_grad():
            current_img = inferer.sample(input_noise=noise, conditioning=art10, mode="concat", diffusion_model=model,
                                         scheduler=scheduler)
            print(f'Pixel range of output_image {step+1} from testing_set: {torch.min(current_img)}, {torch.max(current_img)}')
            print(f'Shape of current_img: {current_img.shape}')
            # print(f'Pixel range of output_image {step+1} from validation_set: {torch.min(current_img)}, {torch.max(current_img)}')
            # Linearly scale from [-1,1] to [0,1]
            # art10_scaled = (art10 + 1) / 2
            # pseudoart100_scaled = (pseudoart100 + 1) / 2
            # current_img_scaled = (current_img + 1) / 2
            # print(f'Pixel range art10_scaled: {torch.min(art10_scaled)}, {torch.max(art10_scaled)}')
            # print(f'Pixel range pseudoart100_scaled: {torch.min(pseudoart100_scaled)}, {torch.max(pseudoart100_scaled)}')
            # print(f'Pixel range current_img_scaled: {torch.min(current_img_scaled)}, {torch.max(current_img_scaled)}')

            # Save model outputs as .tiff and .png images
            output_array = current_img[0, 0, 8:-8, :].cpu().numpy()
            output_array = (output_array * 255).clip(0, 255).astype(np.uint8)
            # Image.fromarray(output_array, mode="L").save(f'{output_dir}/output_{step+1}.tiff')
            Image.fromarray(output_array, mode="L").save(f'{output_dir}/output_{step+1}.png')

            # --- Which location does this sample belong to? ---
            batch_size = current_img.shape[0]
            for i in range(batch_size):
                # global index across the whole loader (works even if batch_size > 1)
                sample_idx = step * batch_size + i
                loc_idx = sample_idx // 10  # 10 B-scans per location
                loc_id = f"{loc_idx:04d}"

                # ART100 ref (float32 0..255), crop to match your 8:-8 vertical slice
                if loc_id not in art100_refs:
                    print(f"⚠️ Missing ART100 ref for loc {loc_id}")
                    continue
                ref_full = art100_refs[loc_id]  # HxW
                ref_t = torch.tensor(ref_full, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                print(f'ART100 pixel range: {ref_t.min()}, {ref_t.max()}')
                ref_t = (ref_t - ref_t.min()) / (ref_t.max() - ref_t.min())
                ref_t = ref_t.cpu()
                print(f'ART100 pixel range after normalisation: {ref_t.min()}, {ref_t.max()}')

                # Generated slice as float for diff (use the tensor directly to keep its native [0,1] range)
                gen_slice = current_img[i, 0, 8:-8, :].cpu().numpy()  # (H-16, W), likely 0..1

                # Save difference map (Generated - ART100), using bwr
                diff_dir = os.path.join(output_dir, "difference_maps", f"loc_{loc_id}")
                diff_path = os.path.join(diff_dir, f"diff_bscan_{sample_idx + 1:03d}.png")

                # Pick color span based on range: if your net outputs 0..1, ±0.2..±0.5 works well; if 0..255, use ±40..±60
                # This auto-chooses a decent default:
                vspan = 0.5 if gen_slice.max() <= 1.1 else 40.0

                save_diff_map_bwr_fixed(gen_slice, ref_full, diff_path)
            
            writer.add_image(f'Testing/Sample', current_img[0, :, 8:-8, :], epoch + 1)
            #writer.add_image(f'Testing_valid_region/Sample_valid_region', output_valid_region.squeeze(0), epoch + 1)
            #writer.add_image(f'Testing_valid_region/Sample_masking_input_artifacts', output_masking_input_artifacts.squeeze(0), epoch + 1)
            writer.add_image(f'Testing/Input', art10[0, :, 8:-8, :], epoch + 1)
            writer.add_image(f'Testing/Target', pseudoart100[0, :, 8:-8, :], epoch + 1)
            #writer.add_image(f'Testing_valid_region/Target_masking_input_artifacts', input_region_mask[0, :, :, :]*pseudoart100_scaled[0, :, 8:-8, :], epoch + 1)

            current_img = current_img.cpu()
            # pseudoart100 = pseudoart100.cpu()
            art10 = art10.cpu()
            ## Compute MSE between for the current batch
            # mse_batch = mean_flat((current_img[:, :, 8:-8, :] - pseudoart100[:, :, 8:-8, :]) ** 2)
            mse_batch = mean_flat((current_img[:, :, 8:-8, :] - ref_t) ** 2)
            # mse_batch_valid_region = mean_flat((output_valid_region - pseudoart100[:, :, 8:-8, :]) ** 2)
            # mse_batch_masking_input_artifacts = mean_flat((output_masking_input_artifacts - pseudoart100[:, :, 8:-8, :]*input_region_mask) ** 2)

            ## Compute PSNR for the current batch
            # psnr_batch = PSNR(current_img[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])
            psnr_batch = PSNR(current_img[:, :, 8:-8, :], ref_t)
            # psnr_batch_valid_region = PSNR(output_valid_region, pseudoart100[:, :, 8:-8, :])
            # psnr_batch_masking_input_artifacts = PSNR(output_masking_input_artifacts, pseudoart100[:, :, 8:-8, :]*input_region_mask)

            ## Compute MY PSNR for the current batch
            # my_psnr_batch = psnr(current_img[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])
            my_psnr_batch = psnr(current_img[:, :, 8:-8, :], ref_t)
            # my_psnr_batch_valid_region = psnr(output_valid_region, pseudoart100[:, :, 8:-8, :])
            # my_psnr_batch_masking_input_artifacts = psnr(output_masking_input_artifacts, pseudoart100[:, :, 8:-8, :]*input_region_mask)

            ## Compute SSIM for the current batch
            # ssim_batch = SSIM(current_img[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])
            ssim_batch = SSIM(current_img[:, :, 8:-8, :], ref_t)
            # ssim_batch_valid_region = SSIM(output_valid_region, pseudoart100[:, :, 8:-8, :])
            # ssim_batch_masking_input_artifacts = SSIM(output_masking_input_artifacts, pseudoart100[:, :, 8:-8, :]*input_region_mask)

            ## Compute LPIPS for the current batch
            # lpips_batch = LPIPS(current_img[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])
            lpips_batch = LPIPS(current_img[:, :, 8:-8, :], ref_t)
            # lpips_batch_valid_region = LPIPS(output_valid_region, pseudoart100[:, :, 8:-8, :])
            # lpips_batch_masking_input_artifacts = LPIPS(output_masking_input_artifacts, pseudoart100[:, :, 8:-8, :]*input_region_mask)

            ## Compute LPIPS RAD-IMAGENET for the current batch
            # lpips_rad_batch = LPIPS_RAD(current_img[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])
            lpips_rad_batch = LPIPS_RAD(current_img[:, :, 8:-8, :], ref_t)
            # lpips_rad_batch_valid_region = LPIPS_RAD(output_valid_region, pseudoart100[:, :, 8:-8, :])
            # lpips_rad_batch_masking_input_artifacts = LPIPS_RAD(output_masking_input_artifacts, pseudoart100[:, :, 8:-8, :]*input_region_mask)

            # Compute PSEUDO-LPIPS for the current batch
            # pseudo_lpips_batch = LPIPS_RESNET(current_img[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])
            pseudo_lpips_batch = LPIPS_RESNET(current_img[:, :, 8:-8, :], ref_t)
            # pseudo_lpips_batch_valid_region = PSEUDO_LPIPS(output_valid_region, pseudoart100[:, :, 8:-8, :])
            # pseudo_lpips_batch_masking_input_artifacts = PSEUDO_LPIPS(output_masking_input_artifacts, pseudoart100[:, :, 8:-8, :]*input_region_mask)

            ## Append the score of the current batch to the corresponding list
            # MSE
            mse_batches.append(mse_batch.mean().cpu().numpy())
            # mse_batches_valid_region.append(mse_batch_valid_region.mean().cpu())
            # mse_batches_masking_input_artifacts.append(mse_batch_masking_input_artifacts.mean().cpu())

            # PSNR
            psnr_batches.append(psnr_batch.cpu())
            # psnr_batches_valid_region.append(psnr_batch_valid_region.cpu())
            # psnr_batches_masking_input_artifacts.append(psnr_batch_masking_input_artifacts.cpu())

            # MY PSNR
            my_psnr_batches.append(my_psnr_batch)
            # my_psnr_batches_valid_region.append(my_psnr_batch_valid_region)
            # my_psnr_batches_masking_input_artifacts.append(my_psnr_batch_masking_input_artifacts)

            # SSIM
            ssim_batches.append(ssim_batch.cpu())
            # ssim_batches_valid_region.append(ssim_batch_valid_region.cpu())
            # ssim_batches_masking_input_artifacts.append(ssim_batch_masking_input_artifacts.cpu())

            # LPIPS
            lpips_batches.append(lpips_batch.detach().cpu().numpy())
            # lpips_batches_valid_region.append(lpips_batch_valid_region.cpu())
            # lpips_batches_masking_input_artifacts.append(lpips_batch_masking_input_artifacts.cpu())

            # LPIPS RAD-IMAGENET
            lpips_rad_batches.append(lpips_rad_batch.detach().cpu().numpy())
            # lpips_rad_batches_valid_region.append(lpips_rad_batch_valid_region.cpu())
            # lpips_rad_batches_masking_input_artifacts.append(lpips_rad_batch_masking_input_artifacts.cpu())

            # PSEUDO-LPIPS
            pseudo_lpips_batches.append(pseudo_lpips_batch.detach().cpu().numpy())
            # pseudo_lpips_batches_valid_region.append(pseudo_lpips_batch_valid_region.cpu())
            # pseudo_lpips_batches_masking_input_artifacts.append(pseudo_lpips_batch_masking_input_artifacts.cpu())

            # We compute and save the difference map between output and target (in the range [0,1])
            
            # save_difference_maps_diffusion_paper(
            #    art10[..., 8:-8, :], pseudoart100[..., 8:-8, :], current_img[..., 8:-8, :],
            #    epoch+1, step + 1,
            #    histogram=True,
            #    writer=writer,
            #    phase='Testing',
            #    folder='/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/image_to_image_translation/results/7th_run/new_acquired_data/art1_inference',
            #)
            
            index += 1

            metrics_per_image.append({
                "image_index": step+1,
                "mse": mse_batch.mean().cpu().numpy(),
                "psnr": psnr_batch.item(),
                "ssim": ssim_batch[0][0].item(),
                "lpips_oct": lpips_batch.item(),
                "lpips_rad": lpips_rad_batch.item(),
                "lpips_resnet": pseudo_lpips_batch.item(),
                # Add other scores as needed
            })
            
            print(
                f"Batch {step+1}: PSNR: {psnr_batch.item()} | SSIM: {ssim_batch.item()} | MSE: {mse_batch.mean().item()} | LPIPS_OCT: {lpips_batch.detach().cpu().numpy()} | LPIPS_RAD: {lpips_rad_batch.detach().cpu().numpy()} | LPIPS_RESNET: {pseudo_lpips_batch.detach().cpu().numpy()}")

            # Instead of using distributed gathering, directly collect samples
            all_images.append(current_img[:, :, 8:-8, :].cpu().numpy())

            # After the loop ends, concatenate all collected images
            print(f"created {len(all_images)} samples")
            arr = np.concatenate(all_images, axis=0)

            # Optionally limit the number of samples if needed
            # arr = arr[:170]  # Uncomment if you want to limit to 170 samples

# here we save all the output images into a .npz file
out_path = os.path.join(output_dir, f"Testing_samples_epoch_{epoch+1}.npz")
print(f"saving samples in the range [0,1] to {out_path}")
np.savez_compressed(out_path, arr)

## Transform each list into a numpy array to then compute mean and std

# PSNR
psnr_batches = np.asarray(psnr_batches, dtype=np.float32)
# psnr_batches_valid_region = np.asarray(psnr_batches_valid_region, dtype=np.float32)
# psnr_batches_masking_input_artifacts = np.asarray(psnr_batches_masking_input_artifacts, dtype=np.float32)

# MY PSNR
my_psnr_batches = np.asarray(my_psnr_batches, dtype=np.float32)
# my_psnr_batches_valid_region = np.asarray(my_psnr_batches_valid_region, dtype=np.float32)
# my_psnr_batches_masking_input_artifacts = np.asarray(my_psnr_batches_masking_input_artifacts, dtype=np.float32)

# SSIM
ssim_batches = np.asarray(ssim_batches, dtype=np.float32)
# ssim_batches_valid_region = np.asarray(ssim_batches_valid_region, dtype=np.float32)
# ssim_batches_masking_input_artifacts = np.asarray(ssim_batches_masking_input_artifacts, dtype=np.float32)

# MSE
mse_batches = np.asarray(mse_batches, dtype=np.float32)
# mse_batches_valid_region = np.asarray(mse_batches_valid_region, dtype=np.float32)
# mse_batches_masking_input_artifacts = np.asarray(mse_batches_masking_input_artifacts, dtype=np.float32)

# LPIPS
lpips_batches = np.asarray(lpips_batches, dtype=np.float32)
# lpips_batches_valid_region = np.asarray(lpips_batches_valid_region, dtype=np.float32)
# lpips_batches_masking_input_artifacts = np.asarray(lpips_batches_masking_input_artifacts, dtype=np.float32)

# LPIPS RAD-IMAGENET
lpips_rad_batches = np.asarray(lpips_rad_batches, dtype=np.float32)
# lpips_rad_batches_valid_region = np.asarray(lpips_rad_batches_valid_region, dtype=np.float32)
# lpips_rad_batches_masking_input_artifacts = np.asarray(lpips_rad_batches_masking_input_artifacts, dtype=np.float32)

# PSEUDO-LPIPS
pseudo_lpips_batches = np.asarray(pseudo_lpips_batches, dtype=np.float32)
# pseudo_lpips_batches_valid_region = np.asarray(pseudo_lpips_batches_valid_region, dtype=np.float32)
# pseudo_lpips_batches_masking_input_artifacts = np.asarray(pseudo_lpips_batches_masking_input_artifacts, dtype=np.float32)

## Calculate mean and std over the whole test set

# PSNR
avg_psnr, std_psnr = np.mean(psnr_batches), np.std(psnr_batches)
# avg_psnr_valid_region, std_psnr_valid_region = np.mean(psnr_batches_valid_region), np.std(psnr_batches_valid_region)
# avg_psnr_masking_input_artifacts, std_psnr_masking_input_artifacts = np.mean(psnr_batches_masking_input_artifacts), np.std(psnr_batches_masking_input_artifacts)

# MY PSNR
avg_my_psnr, std_my_psnr = np.mean(my_psnr_batches), np.std(my_psnr_batches)
# avg_my_psnr_valid_region, std_my_psnr_valid_region = np.mean(my_psnr_batches_valid_region), np.std(my_psnr_batches_valid_region)
# avg_my_psnr_masking_input_artifacts, std_my_psnr_masking_input_artifacts = np.mean(my_psnr_batches_masking_input_artifacts), np.std(my_psnr_batches_masking_input_artifacts)

# SSIM
avg_ssim, std_ssim = np.mean(ssim_batches), np.std(ssim_batches)
# avg_ssim_valid_region, std_ssim_valid_region = np.mean(ssim_batches_valid_region), np.std(ssim_batches_valid_region)
# avg_ssim_masking_input_artifacts, std_ssim_masking_input_artifacts = np.mean(ssim_batches_masking_input_artifacts), np.std(ssim_batches_masking_input_artifacts)

# MSE
avg_mse, std_mse = np.mean(mse_batches), np.std(mse_batches)
# avg_mse_valid_region, std_mse_valid_region = np.mean(mse_batches_valid_region), np.std(mse_batches_valid_region)
# avg_mse_masking_input_artifacts, std_mse_masking_input_artifacts = np.mean(mse_batches_masking_input_artifacts), np.std(mse_batches_masking_input_artifacts)

# LPIPS
avg_lpips, std_lpips = np.mean(lpips_batches), np.std(lpips_batches)
# avg_lpips_valid_region, std_lpips_valid_region = np.mean(lpips_batches_valid_region), np.std(lpips_batches_valid_region)
# avg_lpips_masking_input_artifacts, std_lpips_masking_input_artifacts = np.mean(lpips_batches_masking_input_artifacts), np.std(lpips_batches_masking_input_artifacts)

# LPIPS RAD
avg_lpips_rad, std_lpips_rad = np.mean(lpips_rad_batches), np.std(lpips_rad_batches)
# avg_lpips_rad_valid_region, std_lpips_rad_valid_region = np.mean(lpips_rad_batches_valid_region), np.std(lpips_rad_batches_valid_region)
# avg_lpips_rad_masking_input_artifacts, std_lpips_rad_masking_input_artifacts = np.mean(lpips_rad_batches_masking_input_artifacts), np.std(lpips_rad_batches_masking_input_artifacts)

# PSEUDO-LPIPS
avg_pseudo_lpips, std_pseudo_lpips = np.mean(pseudo_lpips_batches), np.std(pseudo_lpips_batches)
# avg_pseudo_lpips_valid_region, std_pseudo_lpips_valid_region = np.mean(pseudo_lpips_batches_valid_region), np.std(pseudo_lpips_batches_valid_region)
# avg_pseudo_lpips_masking_input_artifacts, std_pseudo_lpips_masking_input_artifacts = np.mean(pseudo_lpips_batches_masking_input_artifacts), np.std(pseudo_lpips_batches_masking_input_artifacts)

df_metrics = pd.DataFrame(metrics_per_image)
df_metrics.to_csv(f"{output_dir}/ddpm_test_metrics.csv", index=False)
print(f"Saved individual test scores to {output_dir}/ddpm_test_metrics.csv")

# Log average metrics to TensorBoard
metrics_summary = {
    "PSNR": avg_psnr,
    "SSIM": avg_ssim,
    "MSE": avg_mse,
    "LPIPS_OCT": avg_lpips,
    "LPIPS_RAD": avg_lpips_rad,
    "LPIPS_RESNET": avg_pseudo_lpips,
}

for metric_name, value in metrics_summary.items():
    writer.add_scalar(f"Testing_metrics/{metric_name}", value.item(), epoch + 1)"""

# ==============================
# 🟡 BASELINE METRIC COMPUTATION (ART10 vs ART100)
# ==============================
print("\n--- Computing baseline metrics between input ART images and ART100 references ---")

baseline_mse, baseline_psnr, baseline_ssim = [], [], []
baseline_lpips, baseline_lpips_rad = [], []

# Loop through the test loader again (same order)
for step, (art_input, _, _) in enumerate(test_loader):
    art_input = art_input.to(device)

    # Determine which location this belongs to (same as before)
    loc_idx = step // 10
    loc_id = f"{loc_idx:04d}"
    if loc_id not in art100_refs:
        print(f"⚠️ Missing ART100 ref for loc {loc_id}")
        continue

    ref_full = art100_refs[loc_id]
    ref_t = torch.tensor(ref_full, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    ref_t = (ref_t - ref_t.min()) / (ref_t.max() - ref_t.min())
    print(f'ref_t pixel range: {ref_t.min()}, {ref_t.max()}')
    print(f'ref_t size: {ref_t.shape}')

    with torch.no_grad():
        # Crop to match the 8:-8 region
        art_crop = art_input[:, :, 8:-8, :]
        print(f'art_crop pixel range: {art_crop.min()}, {art_crop.max()}')
        print(f'art_crop size: {art_crop.shape}')

        # Compute all metrics (same normalization and objects as above)
        mse_b = mean_flat((art_crop - ref_t) ** 2).mean().cpu().numpy()
        psnr_b = PSNR(art_crop, ref_t).cpu().item()
        ssim_b = SSIM(art_crop, ref_t).cpu().numpy().mean()

        lpips_b = LPIPS(art_crop, ref_t).detach().cpu().numpy()
        lpips_rad_b = LPIPS_RAD(art_crop, ref_t).detach().cpu().numpy()

    baseline_mse.append(mse_b)
    baseline_psnr.append(psnr_b)
    baseline_ssim.append(ssim_b)
    baseline_lpips.append(lpips_b)
    baseline_lpips_rad.append(lpips_rad_b)

# Convert to arrays
baseline_mse = np.array(baseline_mse, dtype=np.float32)
baseline_psnr = np.array(baseline_psnr, dtype=np.float32)
baseline_ssim = np.array(baseline_ssim, dtype=np.float32)
baseline_lpips = np.array(baseline_lpips, dtype=np.float32)
baseline_lpips_rad = np.array(baseline_lpips_rad, dtype=np.float32)

# Compute average ± std
b_avg_mse, b_std_mse = baseline_mse.mean(), baseline_mse.std()
b_avg_psnr, b_std_psnr = baseline_psnr.mean(), baseline_psnr.std()
b_avg_ssim, b_std_ssim = baseline_ssim.mean(), baseline_ssim.std()
b_avg_lpips, b_std_lpips = baseline_lpips.mean(), baseline_lpips.std()
b_avg_lpips_rad, b_std_lpips_rad = baseline_lpips_rad.mean(), baseline_lpips_rad.std()

# Save baseline results
df_baseline = pd.DataFrame({
    "MSE": [b_avg_mse],
    "MSE_std": [b_std_mse],
    "PSNR": [b_avg_psnr],
    "PSNR_std": [b_std_psnr],
    "SSIM": [b_avg_ssim],
    "SSIM_std": [b_std_ssim],
    "LPIPS_OCT": [b_avg_lpips],
    "LPIPS_OCT_std": [b_std_lpips],
    "LPIPS_RAD": [b_avg_lpips_rad],
    "LPIPS_RAD_std": [b_std_lpips_rad],
})
df_baseline.to_csv(f"{output_dir}/baseline_metrics.csv", index=False)

print(f"\n✅ Baseline metrics saved to {output_dir}/baseline_metrics.csv")
print(
    f"Baseline metrics (ART input vs ART100): "
    f"PSNR: {b_avg_psnr:.4f} ± {b_std_psnr:.4f} | "
    f"SSIM: {b_avg_ssim:.4f} ± {b_std_ssim:.4f} | "
    f"MSE: {b_avg_mse:.4f} ± {b_std_mse:.4f} | "
    f"LPIPS: {b_avg_lpips:.4f} ± {b_std_lpips:.4f} | "
    f"LPIPS_RAD: {b_avg_lpips_rad:.4f} ± {b_std_lpips_rad:.4f}"
)

# print(
#    f"Testing metrics, epoch {epoch + 1}: PSNR: {avg_psnr.item():.4f} ± {std_psnr.item():.4f} | SSIM: {avg_ssim.item():.4f} ± {std_ssim.item():.4f} | MSE: {avg_mse.item():.4f} ± {std_mse.item():.4f} | LPIPS_OCT: {avg_lpips.item():.4f} ± {std_lpips.item():.4f} | LPIPS_RAD: {avg_lpips_rad.item():.4f} ± {std_lpips_rad.item():.4f} | LPIPS_RESNET: {avg_pseudo_lpips.item():.4f} ± {std_pseudo_lpips.item():.4f}")
"""
print(
    f"Testing metrics, epoch {epoch + 1}, valid region only: PSNR: {avg_psnr_valid_region.item():.4f} ± {std_psnr_valid_region.item():.4f} | MY_PSNR: {avg_my_psnr_valid_region.item():.4f} ± {std_my_psnr_valid_region.item():.4f} | SSIM: {avg_ssim_valid_region.item():.4f} ± {std_ssim_valid_region.item():.4f} | MSE: {avg_mse_valid_region.item():.4f} ± {std_mse_valid_region.item():.4f} | LPIPS: {avg_lpips_valid_region.item():.4f} ± {std_lpips_valid_region.item():.4f} | LPIPS_RAD: {avg_lpips_rad_valid_region.item():.4f} ± {std_lpips_rad_valid_region.item():.4f} | PSEUDO_LPIPS: {avg_pseudo_lpips_valid_region.item():.4f} ± {std_pseudo_lpips_valid_region.item():.4f}")
print(
    f"Testing metrics, epoch {epoch + 1}, masking input artifacts: PSNR: {avg_psnr_masking_input_artifacts.item():.4f} ± {std_psnr_masking_input_artifacts.item():.4f} | MY_PSNR: {avg_my_psnr_masking_input_artifacts.item():.4f} ± {std_my_psnr_masking_input_artifacts.item():.4f} | SSIM: {avg_ssim_masking_input_artifacts.item():.4f} ± {std_ssim_masking_input_artifacts.item():.4f} | MSE: {avg_mse_masking_input_artifacts.item():.4f} ± {std_mse_masking_input_artifacts.item():.4f} | LPIPS: {avg_lpips_masking_input_artifacts.item():.4f} ± {std_lpips_masking_input_artifacts.item():.4f} | LPIPS_RAD: {avg_lpips_rad_masking_input_artifacts.item():.4f} ± {std_lpips_rad_masking_input_artifacts.item():.4f} PSEUDO_LPIPS: {avg_pseudo_lpips_masking_input_artifacts.item():.4f} ± {std_pseudo_lpips_masking_input_artifacts.item():.4f}")
"""