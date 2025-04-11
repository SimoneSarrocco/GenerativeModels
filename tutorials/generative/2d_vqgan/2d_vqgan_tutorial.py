# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
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

# %% [markdown]
# # Vector Quantized Generative Adversarial Networks with MedNIST Dataset
#
# This tutorial illustrates how to use MONAI for training a Vector Quantized Generative Adversarial Network (VQGAN) on 2D images.
#
#
# ## Setup environment

# %%
# !python -c "import monai" || pip install -q "monai-weekly[tqdm]"
# !python -c "import matplotlib" || pip install -q matplotlib
# %matplotlib inline

# %% [markdown]
# ## Setup imports

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# %%
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.utils import first, set_determinism
from torch.nn import L1Loss
from tqdm import tqdm
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import VQVAE, PatchDiscriminator
from generative.metrics.ssim import SSIMMetric
from dataset import OCTDataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import argparse
print_config()


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Train VQGAN model")
    parser.add_argument("--num_embeddings", type=int, required=True, help="Number of embedding vectors")
    parser.add_argument("--embedding_dimension", type=int, required=True, help="Embedding dimension of each vector")
    parser.add_argument("--pixel_range", type=int, default=0, help="Pixel range of each image, 0 means [0,1], -1 means [-1,1]")
    parser.add_argument("--iteration", type=int, default=0, help="Number of current epoch from which to sample")
    parser.add_argument("--model_name", type=str, default='', help="Name of the model configuration")

    args = parser.parse_args()

    return args


def save_checkpoint(model, epoch, save_dir="checkpoints"):
    """
    Saves the model in the .ckpt format required by BBDM.

    Args:
        model: The trained VQ-GAN model.
        epoch: Current epoch number.
        save_dir: Directory where checkpoints will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    checkpoint = {
        "state_dict": model.state_dict(),  # BBDM expects this format
        "epoch": epoch,  # Save the current epoch for reference
    }

    checkpoint_path = os.path.join(save_dir, f"vqgan_best_checkpoint.ckpt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

# %% [markdown]
# ## Setup data directory
#
# You can specify a directory with the MONAI_DATA_DIRECTORY environment variable.
#
# This allows you to save results and reuse downloads.
#
# If not specified a temporary directory will be used.

# %%
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

# %% [markdown]
# ## Set deterministic training for reproducibility

# %%
set_determinism(1927)

# %% [markdown]
# ## Setup MedNIST Dataset and training and validation dataloaders
# In this tutorial, we will train our models on the MedNIST dataset available on MONAI
# (https://docs.monai.io/en/stable/apps.html#monai.apps.MedNISTDataset). In order to train faster, we will select just
# one of the available classes ("HeadCT").

args = parse_args_and_config()

train = np.load('/home/simone.sarrocco/thesis/project/data/train_set_patient_split.npz')['images']
val = np.load('/home/simone.sarrocco/thesis/project/data/val_set_patient_split.npz')['images']
test = np.load('/home/simone.sarrocco/thesis/project/data/test_set_patient_split.npz')['images']

train_images = []
val_images = []
test_images = []

for i in range(len(train)):
    art10 = torch.tensor(train[i, :1, ...])
    pseudoart100 = torch.tensor(train[i, -1:, ...])
    train_images.append(art10)
    train_images.append(pseudoart100)
train_images = torch.stack(train_images, 0)

for i in range(len(val)):
    art10 = torch.tensor(val[i, :1, ...])
    pseudoart100 = torch.tensor(val[i, -1:, ...])
    val_images.append(art10)
    val_images.append(pseudoart100)
val_images = torch.stack(val_images, 0)

for i in range(len(test)):
    art10 = torch.tensor(test[i, :1, ...])
    pseudoart100 = torch.tensor(test[i, -1:, ...])
    test_images.append(art10)
    test_images.append(pseudoart100)
test_images = torch.stack(test_images, 0)

val_images = torch.cat((val_images, test_images), dim=0)

train_data = OCTDataset(train_images, transform=True, pixel_range=args.pixel_range)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=64)
print(f'Shape of training set: {train_images.shape}')
# train_datalist = [{"image": train[i, -1:, ...]} for i in range(len(train))]

# %%
val_data = OCTDataset(val_images)
# val_datalist = [{"image": val[i, -1:, ...]} for i in range(len(val))]
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=64)
print(f'Shape of validation set: {val_images.shape}')


# %%
device = torch.device("cuda")
model = VQVAE(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(256, 512),
    num_res_channels=512,
    num_res_layers=2,
    downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
    upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
    num_embeddings=args.num_embeddings,  # this is "k"
    embedding_dim=args.embedding_dimension,  # this is "d"
)
model.to(device)


discriminator = PatchDiscriminator(spatial_dims=2, in_channels=1, num_layers_d=3, num_channels=64)
discriminator.to(device)

perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex", device='cuda')
perceptual_loss.to(device)

optimizer_g = torch.optim.Adam(params=model.parameters(), lr=1e-4)
optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=5e-4)

# %%
l1_loss = L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")
adv_weight = 0.01
perceptual_weight = 0.001

# tensorboard_dir = f'/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/2d_vqgan/vqgan-2layers-embeddim-{args.embedding_dimension}-num-embed-{args.num_embeddings}-pixel_range-{args.pixel_range}/tensorboard_log'
tensorboard_dir = f'/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/2d_vqgan/{args.model_name}/tensorboard_log'
writer = SummaryWriter(log_dir=tensorboard_dir)
# Define the directory to save checkpoints
# checkpoint_dir = f"/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/2d_vqgan/vqgan-2layers-embeddim-{args.embedding_dimension}-num-embed-{args.num_embeddings}-pixel_range-{args.pixel_range}/checkpoints"
checkpoint_dir = f"/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/2d_vqgan/{args.model_name}/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# %% [markdown]
# ### Model training
# Here, we are training our model for 100 epochs (training time: ~50 minutes).

# %%
n_epochs = 100
val_interval = 1
epoch_recon_loss_list = []
epoch_gen_loss_list = []
epoch_disc_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []
n_example_images = 1
early_stopper = EarlyStopper(patience=20, min_delta=0.0001)

total_start = time.time()
i = 0
best_val_loss = float("inf")
PSNR = PeakSignalNoiseRatio().to(device)
# SSIM = StructuralSimilarityIndexMeasure().to(device)
SSIM = SSIMMetric(spatial_dims=2, reduction='mean_batch')
# LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

for epoch in range(n_epochs):
    model.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    mse_batches, psnr_batches, ssim_batches = [], [], []
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    # Training loop
    for step, batch in progress_bar:
        images = batch.to(device)
        optimizer_g.zero_grad(set_to_none=True)

        # Generator part
        reconstruction, quantization_loss = model(images=images)

        i += 1
        logits_fake = discriminator(reconstruction.contiguous().float())[-1]

        recons_loss = l1_loss(reconstruction.float(), images.float())
        p_loss = perceptual_loss(reconstruction.float(), images.float())
        generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
        loss_g = recons_loss + quantization_loss + perceptual_weight * p_loss + adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()

        # Discriminator part
        optimizer_d.zero_grad(set_to_none=True)

        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = discriminator(images.contiguous().detach())[-1]
        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = adv_weight * discriminator_loss

        loss_d.backward()
        optimizer_d.step()

        epoch_loss += recons_loss.item()
        gen_epoch_loss += generator_loss.item()
        disc_epoch_loss += discriminator_loss.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
            }
        )
        with torch.no_grad():
            # Log all the training losses to tensorboard
            writer.add_scalar('Loss/perceptual_loss', p_loss, i)
            writer.add_scalar('Loss/generator_loss', generator_loss, i)
            writer.add_scalar('Loss/recons_loss', recons_loss, i)
            writer.add_scalar('Loss/total_loss_g', loss_g, i)
            writer.add_scalar('Loss/discriminator_loss', loss_d, i)

            # Every epoch visualize input image and corresponding reconstruction on tensorboard
            if i % 1980 == 0:
                if args.pixel_range == -1:
                    writer.add_image(tag=f'Training/Input',
                                     # img_tensor=images[:n_example_images, 0, 8:-8, :],
                                     img_tensor=(images[:n_example_images, 0, 8:-8, :]+1)/2,
                                     global_step=i)
                    writer.add_image(tag=f'Training/Output', img_tensor=(reconstruction[:n_example_images, 0, 8:-8, :]+1)/2,
                                     global_step=i)
                elif args.pixel_range == 0:
                    writer.add_image(tag=f'Training/Input',
                                     # img_tensor=images[:n_example_images, 0, 8:-8, :],
                                     img_tensor=images[:n_example_images, 0, 8:-8, :],
                                     global_step=i)
                    writer.add_image(tag=f'Training/Output', img_tensor=reconstruction[:n_example_images, 0, 8:-8, :],
                                     global_step=i)

    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
    epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

    # Validation loop
    model.eval()
    val_loss = 0
    mse_batches, psnr_batches, ssim_batches, perceptual_batches = [], [], [], []
    with torch.no_grad():
        for val_step, batch in enumerate(val_loader, start=1):
            images = batch.to(device)

            # reconstruction, quantization_loss = model(images=images)
            reconstruction, quantization_loss = model(images=images)

            # get the first sample from the first validation batch for visualization
            # purposes
            if val_step == 1:
                if args.pixel_range == -1:
                    intermediary_images.append((reconstruction[:n_example_images, 0]+1)/2)
                    # writer.add_image(tag=f'Validation/Input', img_tensor=images[:n_example_images, 0], global_step=i)
                    writer.add_image(tag=f'Validation/Input', img_tensor=(images[:n_example_images, 0, 8:-8, :]+1)/2, global_step=i)
                    writer.add_image(tag=f'Validation/Output', img_tensor=(reconstruction[:n_example_images, 0, 8:-8, :]+1)/2,
                                     global_step=i)
                elif args.pixel_range == 0:
                    intermediary_images.append(reconstruction[:n_example_images, 0])
                    # writer.add_image(tag=f'Validation/Input', img_tensor=images[:n_example_images, 0], global_step=i)
                    writer.add_image(tag=f'Validation/Input', img_tensor=images[:n_example_images, 0, 8:-8, :], global_step=i)
                    writer.add_image(tag=f'Validation/Output', img_tensor=reconstruction[:n_example_images, 0, 8:-8, :],
                                     global_step=i)

            recons_loss = l1_loss(reconstruction.float(), images.float())

            val_loss += recons_loss.item()

            # Compute PSNR, SSIM, and MSE between input and reconstructed image
            mse_batch = mean_flat((reconstruction[:, :, 8:-8, :] - images[:, :, 8:-8, :]) ** 2)
            psnr_batch = PSNR(reconstruction[:, :, 8:-8, :], images[:, :, 8:-8, :])
            ssim_batch = SSIM._compute_metric(reconstruction[:, :, 8:-8, :], images[:, :, 8:-8, :])
            # perceptual_batch = perceptual_loss(reconstruction[:, :, 8:-8, :].float(), images[:, :, 8:-8, :].float())

            mse_batches.append(mse_batch.mean().cpu())
            psnr_batches.append(psnr_batch.cpu())
            ssim_batches.append(ssim_batch.cpu())
            # perceptual_batches.append(perceptual_batch.cpu())

        val_loss /= val_step
        val_recon_epoch_loss_list.append(val_loss)
        writer.add_scalar('Loss/val_loss', val_loss, epoch+1)

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

    # Save checkpoint if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, epoch + 1, save_dir=checkpoint_dir)
        print(f"Saved best model with validation loss: {val_loss:.6f}")

    # Early Stopping if the validation loss has not decreased for more than value_of_patience epochs in a row
    if early_stopper.early_stop(val_loss):
        print(f"Early stopping at epoch: {epoch+1}")
        break

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")
# %% [markdown]
# ### Cleanup data directory
#
# Remove directory if a temporary was used.

# %%
if directory is None:
    shutil.rmtree(root_dir)

# Load checkpoint
ckpt_path = "/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/2d_vqgan/{args.model_name}/checkpoints/vqgan_best_checkpoint.ckpt"
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Estimate latent range
z_min, z_max = float("inf"), -float("inf")

with torch.no_grad():
    for i, batch in enumerate(train_loader):
        images = batch.to(device)
        z = model.encoder(images)
        if isinstance(z, tuple):  # handle VQGAN returning (quant, loss)
            z = z[0]
        z_min = min(z_min, z.min().item())
        z_max = max(z_max, z.max().item())

print(f"Estimated latent range: min={z_min:.3f}, max={z_max:.3f}")


tensorboard_dir = '/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/2d_vqgan/denoising_model/tensorboard_log'
writer = SummaryWriter(log_dir=tensorboard_dir)
# Define the directory to save checkpoints
checkpoint_dir = "/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/2d_vqgan/denoising_model/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
PSNR = PeakSignalNoiseRatio().to(device)
# SSIM = StructuralSimilarityIndexMeasure().to(device)
SSIM = SSIMMetric(spatial_dims=2, reduction='mean_batch')
# LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
n_example_images = 1

# Testing loop
with torch.no_grad():
    mse_batches, psnr_batches, ssim_batches = [], [], []
    for val_step, batch in enumerate(val_loader):
        images = batch.to(device)
        # reconstruction, quantization_loss = model(images=images)
        reconstruction, quantization_loss = model(images=images)

        # get the first sample from the first validation batch for visualization
        # purposes
        if val_step == 1:
            if args.pixel_range == -1:
                # writer.add_image(tag=f'Validation/Input', img_tensor=images[:n_example_images, 0], global_step=i)
                writer.add_image(tag=f'Testing/Input', img_tensor=(images[:n_example_images, 0, 8:-8, :] + 1) / 2,
                                 global_step=args.iteration)
                writer.add_image(tag=f'Testing/Output',
                                 img_tensor=(reconstruction[:n_example_images, 0, 8:-8, :] + 1) / 2,
                                 global_step=args.iteration)
            elif args.pixel_range == 0:
                # writer.add_image(tag=f'Validation/Input', img_tensor=images[:n_example_images, 0], global_step=i)
                writer.add_image(tag=f'Testing/Input', img_tensor=images[:n_example_images, 0, 8:-8, :],
                                 global_step=args.iteration)
                writer.add_image(tag=f'Testing/Output', img_tensor=reconstruction[:n_example_images, 0, 8:-8, :],
                                 global_step=args.iteration)

        # Compute PSNR, SSIM, and MSE between input and reconstructed image
        mse_batch = mean_flat((reconstruction[:, :, 8:-8, :] - images[:, :, 8:-8, :]) ** 2)
        psnr_batch = PSNR(reconstruction[:, :, 8:-8, :], images[:, :, 8:-8, :])
        ssim_batch = SSIM._compute_metric(reconstruction[:, :, 8:-8, :], images[:, :, 8:-8, :])
        # perceptual_batch = perceptual_loss(reconstruction[:, :, 8:-8, :].float(), images[:, :, 8:-8, :].float())

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
        writer.add_scalar(f"Testing_metrics/{metric_name}", value.item(), args.iteration)
    print(
        f"Testing metrics, epoch {args.iteration}: PSNR: {avg_psnr.item():.5f} ± {std_psnr.item():.5f} | SSIM: {avg_ssim.item():.5f} ± {std_ssim.item():.5f} | MSE: {avg_mse.item():.5f} ± {std_mse.item():.5f}")
