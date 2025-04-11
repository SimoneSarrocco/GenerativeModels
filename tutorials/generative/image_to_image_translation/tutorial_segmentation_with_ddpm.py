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
from torch.utils.tensorboard import SummaryWriter
from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddpm import DDPMScheduler
from generative.losses import PerceptualLoss
from generative.metrics.ssim import SSIMMetric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


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

set_determinism(42)

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
train = np.load('/home/simone.sarrocco/thesis/project/data/train_set_patient_split.npz')['images']
val = np.load('/home/simone.sarrocco/thesis/project/data/val_set_patient_split.npz')['images']
test = np.load('/home/simone.sarrocco/thesis/project/data/test_set_patient_split.npz')['images']

# train_data_split = torch.tensor(train).view((-1, 1, 496, 768))  # Pass shape as a tuple
# val_data_split = torch.tensor(val).view((-1, 1, 496, 768))  # Pass shape as a tuple
# test_data_split = torch.tensor(test).view((-1, 1, 496, 768))  # Pass shape as a tuple

# final_val_data_split = torch.cat([val_data_split, test_data_split], dim=0)

# train_data = OCTDataset(train_data_split, transform=True)
train_data = OCTDataset(train, transform=True)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=64)
# print(f'Shape of training set: {train_data_split.shape}')
print(f'Shape of training set: {train.shape}')

# val_data = OCTDataset(final_val_data_split)
val_data = OCTDataset(val, transform=False)
# val_datalist = [{"image": val[i, -1:, ...]} for i in range(len(val))]
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=64)
# print(f'Shape of validation set: {val_data_split.shape}')
print(f'Shape of validation set: {val.shape}')


run_number = '4th_run'
#
# ## Define network, scheduler, optimizer, and inferer
#
# At this step, we instantiate the MONAI components to create a DDPM, the UNET, the noise scheduler, and the inferer used for training and sampling. We are using the DDPM scheduler containing 1000 timesteps, and a 2D UNET with attention mechanisms in the 3rd level (`num_head_channels=64`).<br>
#
writer = SummaryWriter(log_dir=f'/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/image_to_image_translation/logs/{run_number}')
device = torch.device("cuda")

checkpoint_dir = f"/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/image_to_image_translation/checkpoints/{run_number}"
os.makedirs(checkpoint_dir, exist_ok=True)

PSNR = PeakSignalNoiseRatio().to(device)
# SSIM = StructuralSimilarityIndexMeasure().to(device)
SSIM = SSIMMetric(spatial_dims=2)
# PERC = PerceptualLoss(spatial_dims=2, device=device)
# LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

resume_training_flag = True  # Set to True to resume training, False to start from scratch
checkpoint_path = "/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/image_to_image_translation/checkpoints/4th_run/ddpm_oct_model_601.pt"

model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=2,
    out_channels=1,
    num_channels=(128, 128, 256, 256, 512, 512),
    attention_levels=(False, False, False, False, False, True),
    num_res_blocks=2,
    num_head_channels=64,
    with_conditioning=False,
)
model.to(device)

scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)
inferer = DiffusionInferer(scheduler)

#
# ### Model training of the Diffusion Model<br>
# We train our diffusion model for 4000 epochs.\
# In every step, we concatenate the original MR image to the noisy segmentation mask, to predict a slightly denoised segmentation mask.\
# This is described in Equation 7 of the paper https://arxiv.org/pdf/2112.03145.pdf.

n_epochs = 2000
val_interval = 1
epoch_loss_list = []
val_epoch_loss_list = []
val_sample = 25
save_interval = 50
# +
validation_samples_path = f'/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/image_to_image_translation/results/{run_number}/validation/output_samples'
os.makedirs(validation_samples_path, exist_ok=True)

scaler = GradScaler('cuda')
total_start = time.time()
i = 0

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

    for step, (art10, pseudoart100) in enumerate(train_loader):
        art10 = art10.to(device)
        pseudoart100 = pseudoart100.to(device)
        # seg = data["label"].to(device)  # this is the ground truth segmentation
        optimizer.zero_grad(set_to_none=True)
        timesteps = torch.randint(0, 1000, (art10.shape[0],)).to(device)  # pick a random time step t

        with autocast('cuda', enabled=True):
            # Generate random noise
            noise = torch.randn_like(pseudoart100).to(device)
            noisy_pseudoart100 = scheduler.add_noise(
                original_samples=pseudoart100, noise=noise, timesteps=timesteps
            )  # we only add noise to the segmentation mask
            combined = torch.cat(
                (art10, noisy_pseudoart100), dim=1
            )  # we concatenate the brain MR image with the noisy segmenatation mask, to condition the generation process
            prediction = model(x=combined, timesteps=timesteps)
            # Get model prediction
            loss = F.mse_loss(prediction.float(), noise.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        i += 1
        writer.add_scalar('Loss/train', loss.item(), i)

    epoch_loss_list.append(epoch_loss / (step + 1))

    """
    if (epoch+1) % val_interval == 0:
    model.eval()
    val_epoch_loss = 0
    for step, (art10, pseudoart100) in enumerate(val_loader):
        art10 = art10.to(device)
        pseudoart100 = pseudoart100.to(device)
        # seg = data_val["label"].to(device)  # this is the ground truth segmentation
        timesteps = torch.randint(0, 1000, (art10.shape[0],)).to(device)
        with torch.no_grad():
            with autocast('cuda', enabled=True):
                noise = torch.randn_like(art10).to(device)
                noisy_pseudoart100 = scheduler.add_noise(original_samples=art10, noise=noise, timesteps=timesteps)
                combined = torch.cat((art10, noisy_pseudoart100), dim=1)
                prediction = model(x=combined, timesteps=timesteps)
                val_loss = F.mse_loss(prediction.float(), noise.float())
        val_epoch_loss += val_loss.item()
    print("Epoch", epoch, "Validation loss", val_epoch_loss / (step + 1))
    writer.add_scalar("Loss/val", val_epoch_loss / (step+1), epoch+1)
    val_epoch_loss_list.append(val_epoch_loss / (step + 1))
    """

    if (epoch+1) % val_sample == 0:
        model.eval()
        mse_batches, psnr_batches, ssim_batches = [], [], []
        for step, (art10, pseudoart100) in enumerate(val_loader):
            if step == 0:
                art10 = art10.to(device)
                pseudoart100 = pseudoart100.to(device)
                timesteps = torch.randint(0, 1000, (art10.shape[0],)).to(device)
                noise = torch.randn_like(art10).to(device)
                current_img = noise  # for the pseudoART100, we start from random noise.
                combined = torch.cat(
                    (art10, noise), dim=1
                )  # We concatenate the input ART10 to add anatomical information.

                scheduler.set_timesteps(num_inference_steps=1000)
                progress_bar = tqdm(scheduler.timesteps)
                chain = torch.zeros(current_img.shape)
                for t in progress_bar:  # go through the noising process
                    with autocast('cuda', enabled=False):
                        with torch.no_grad():
                            model_output = model(combined, timesteps=torch.Tensor((t,)).to(current_img.device))
                            current_img, _ = scheduler.step(
                                model_output, t, current_img
                            )  # this is the prediction x_t at the time step t
                            if t % 100 == 0:
                                chain = torch.cat((chain, current_img.cpu()), dim=-1)
                            combined = torch.cat(
                                (art10, current_img), dim=1
                            )  # in every step during the denoising process, the ART10 is concatenated to add anatomical information
                with torch.no_grad():
                    """
                    if (step + 1) % 5 == 0:
                        plt.style.use("default")
                        plt.imshow(chain[0, 0, ..., 64:].cpu(), vmin=0, vmax=1, cmap="gray")
                        plt.tight_layout()
                        plt.axis("off")
                        plt.savefig(f'{validation_samples_path}/Sample_{step + 1}.png')
                        plt.close()
                    """
                    writer.add_image(f'Validation/Input', art10.squeeze(0), epoch + 1)
                    writer.add_image(f'Validation/Output', current_img.clamp(0., 1.).squeeze(0), epoch + 1)
                    writer.add_image(f'Validation/Target', pseudoart100.squeeze(0), epoch + 1)

                    # Compute PSNR, SSIM, MSE, and LPIPS between target image (pseudoART100) and denoised image (current_img)
                    mse_batch = mean_flat((current_img[:, :, 8:-8, :] - pseudoart100[:, :, 8:-8, :]) ** 2)
                    psnr_batch = PSNR(current_img[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])
                    ssim_batch = SSIM._compute_metric(current_img[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])
                    # perceptual_batch = PERC(current_img[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])

                    mse_batches.append(mse_batch.mean().cpu())
                    psnr_batches.append(psnr_batch.cpu())
                    ssim_batches.append(ssim_batch.cpu())
                    # perceptual_batches.append(perceptual_batch.cpu())

        psnr_batches = np.asarray(psnr_batches, dtype=np.float32)
        ssim_batches = np.asarray(ssim_batches, dtype=np.float32)
        mse_batches = np.asarray(mse_batches, dtype=np.float32)
        # perceptual_batches = np.asarray(perceptual_batches, dtype=np.float32)

        # Calculate averages
        avg_psnr, std_psnr = np.mean(psnr_batches), np.std(psnr_batches)
        avg_ssim, std_ssim = np.mean(ssim_batches), np.std(ssim_batches)
        avg_mse, std_mse = np.mean(mse_batches), np.std(mse_batches)
        # avg_perceptual, std_perceptual = np.mean(perceptual_batches), np.std(perceptual_batches)

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
            f"Validation metrics, epoch {epoch + 1}: PSNR: {avg_psnr.item():.5f} ± {std_psnr.item():.5f} | SSIM: {avg_ssim.item():.5f} ± {std_ssim.item():.5f} | MSE: {avg_mse.item():.5f} ± {std_mse.item():.5f}")

    # Save checkpoint at regular intervals
    if (epoch + 1) % save_interval == 0:
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir)

        # Optionally save loss history
        np.savez(
            f"{checkpoint_dir}/loss_history.npz",
            train_loss=np.array(epoch_loss_list),
            val_loss=np.array(val_epoch_loss_list) if val_epoch_loss_list else np.array([]),
        )

# torch.save(model.state_dict(), f"{checkpoint_dir}/ddpm_oct_model_last_epoch.pt")
save_checkpoint(model, optimizer, scheduler, n_epochs - 1, checkpoint_dir, is_final=True)
total_time = time.time() - total_start
print(f"train diffusion completed, total time: {total_time}.")
plt.style.use("seaborn-bright")
plt.title("Learning Curves Diffusion Model", fontsize=20)
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_loss_list, color="C0", linewidth=2.0, label="Train")
plt.plot(
    np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
    val_epoch_loss_list,
    color="C1",
    linewidth=2.0,
    label="Validation",
)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.savefig('/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/image_to_image_translation/results/losses.png')
plt.close()
# plt.show()
# -

#
# # Sampling of a new segmentation mask for an input image of the validation set<br>
#
# Starting from random noise, we want to generate a segmentation mask for a brain MR image of our validation set.\
# Due to the stochastic generation process, we can sample an ensemble of n different segmentation masks per MR image.\
# First, we pick an image of our validation set, and check the ground truth segmentation mask.
"""
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
