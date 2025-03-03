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
from dataset import OCTDataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
print_config()


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
set_determinism(42)

# %% [markdown]
# ## Setup MedNIST Dataset and training and validation dataloaders
# In this tutorial, we will train our models on the MedNIST dataset available on MONAI
# (https://docs.monai.io/en/stable/apps.html#monai.apps.MedNISTDataset). In order to train faster, we will select just
# one of the available classes ("HeadCT").

# %%
# train_data = MedNISTDataset(root_dir=root_dir, section="training", download=True, progress=False, seed=0)
# train_datalist = [{"image": item["image"]} for item in train_data.data if item["class_name"] == "HeadCT"]

train = np.load('/home/simone.sarrocco/thesis/project/data/train_set_patient_split.npz')['images']
val = np.load('/home/simone.sarrocco/thesis/project/data/val_set_patient_split.npz')['images']
test = np.load('/home/simone.sarrocco/thesis/project/data/test_set_patient_split.npz')['images']
train_art10_images = []
val_art10_images = []
test_art10_images = []
for i in range(len(train)):
    image = torch.tensor(train[i, :1, ...])
    train_art10_images.append(image)
train_art10_images = torch.stack(train_art10_images, 0)

for i in range(len(val)):
    image = torch.tensor(val[i, :1, ...])
    val_art10_images.append(image)
val_art10_images = torch.stack(val_art10_images, 0)

for i in range(len(test)):
    image = torch.tensor(test[i, :1, ...])
    test_art10_images.append(image)
test_art10_images = torch.stack(test_art10_images, 0)

train_data = OCTDataset(train_art10_images)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)
print(f'Shape of training set: {train_art10_images.shape}')
# train_datalist = [{"image": train[i, -1:, ...]} for i in range(len(train))]

# %% [markdown]
# Here we use transforms to augment the training dataset:
#
# 1. `LoadImaged` loads the hands images from files.
# 1. `EnsureChannelFirstd` ensures the original data to construct "channel first" shape.
# 1. `ScaleIntensityRanged` extracts intensity range [0, 255] and scales to [0, 1].
# 1. `RandAffined` efficiently performs rotate, scale, shear, translate, etc. together based on PyTorch affine transform.

# %%
val_data = OCTDataset(val_art10_images)
# val_datalist = [{"image": val[i, -1:, ...]} for i in range(len(val))]
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
print(f'Shape of validation set: {val_art10_images.shape}')

# %% [markdown]
# ### Visualization of the training images

# %%

# %% [markdown]
# ### Define network, scheduler and optimizer
# At this step, we instantiate the MONAI components to create a VQVAE and a Discriminator model. We are using the
# Discriminator to train the Autoencoder with a Generative Adversarial loss, where the VQVAE works as a Generator.
# The VQVAE is trained to minimize the reconstruction error, a perceptual loss using AlexNet as the embedding model
# and an adversarial loss versus the performance of the Discriminator.

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
    num_embeddings=256,
    embedding_dim=32,
)
model.to(device)

discriminator = PatchDiscriminator(spatial_dims=2, in_channels=1, num_layers_d=3, num_channels=64)
discriminator.to(device)

perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")
perceptual_loss.to(device)

optimizer_g = torch.optim.Adam(params=model.parameters(), lr=1e-4)
optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=5e-4)

# %%
l1_loss = L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")
adv_weight = 0.01
perceptual_weight = 0.001

tensorboard_dir = '/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/2d_vqgan/only_art10/tensorboard_log'
writer = SummaryWriter(log_dir=tensorboard_dir)
# Define the directory to save checkpoints
checkpoint_dir = "/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/2d_vqgan/only_art10/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# %% [markdown]
# ### Model training
# Here, we are training our model for 100 epochs (training time: ~50 minutes).

# %%
n_epochs = 500
val_interval = 5
epoch_recon_loss_list = []
epoch_gen_loss_list = []
epoch_disc_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []
n_example_images = 1

total_start = time.time()
i = 0
PSNR = PeakSignalNoiseRatio().to(device)
SSIM = StructuralSimilarityIndexMeasure().to(device)
# LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

for epoch in range(n_epochs):
    model.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    mse_batches, psnr_batches, ssim_batches, perceptual_batches = [], [], [], []
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
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
            if i % 990 == 0:
                writer.add_image(tag=f'Training/Input',
                                 img_tensor=images[:n_example_images, 0, 8:-8, :],
                                 global_step=i)
                writer.add_image(tag=f'Training/Output', img_tensor=reconstruction[:n_example_images, 0, 8:-8, :],
                                 global_step=i)
            """
            # Compute PSNR, SSIM, MSE, and LPIPS between input image and reconstructed image
            mse_batch = mean_flat((reconstruction[:, :, 8:-8, :] - images[:, :, 8:-8, :]) ** 2)
            psnr_batch = PSNR(reconstruction[:, :, 8:-8, :], images[:, :, 8:-8, :])
            ssim_batch = SSIM(reconstruction[:, :, 8:-8, :], images[:, :, 8:-8, :])
            perceptual_batch = perceptual_loss(reconstruction[:, :, 8:-8, :].float(), images[:, :, 8:-8, :].float())

            mse_batches.append(mse_batch.mean().cpu())
            psnr_batches.append(psnr_batch.cpu())
            ssim_batches.append(ssim_batch.cpu())
            perceptual_batches.append(perceptual_batch.cpu())

            # output_3_channels = reconstruction[:, :, 8:-8, :].repeat(1, 3, 1, 1)  # (Batch, Channels, Height, Width)
            # target_3_channels = images[:, :, 8:-8, :].repeat(1, 3, 1, 1)  # (Batch, Channels, Height, Width)
            # lpips_batch = LPIPS(output_3_channels, target_3_channels)
            # lpips_batches.append(lpips_batch.cpu())
            """
    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
    epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))
    """
    psnr_batches = np.asarray(psnr_batches, dtype=np.float32)
    ssim_batches = np.asarray(ssim_batches, dtype=np.float32)
    mse_batches = np.asarray(mse_batches, dtype=np.float32)
    perceptual_batches = np.asarray(perceptual_batches, dtype=np.float32)

    # Calculate averages
    avg_psnr, std_psnr = np.mean(psnr_batches), np.std(psnr_batches)
    avg_ssim, std_ssim = np.mean(ssim_batches), np.std(ssim_batches)
    avg_mse, std_mse = np.mean(mse_batches), np.std(mse_batches)
    avg_perceptual, std_perceptual = np.mean(perceptual_batches), np.std(perceptual_batches)

    # Log average metrics to TensorBoard
    metrics_summary = {
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "MSE": avg_mse,
        "PERC_LOSS": avg_perceptual,
    }

    for metric_name, value in metrics_summary.items():
        writer.add_scalar(f"Training_metrics/{metric_name}", value.item(), epoch + 1)
    print(
        f"Training metrics, epoch {epoch + 1}: PSNR: {avg_psnr.item():.5f} ± {std_psnr.item():.5f} | SSIM: {avg_ssim.item():.5f} ± {std_ssim.item():.5f} | MSE: {avg_mse.item():.5f} ± {std_mse.item():.5f} | PERC_LOSS: {avg_perceptual.item():.5f} ± {std_perceptual.item():.5f}")
    """
    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_loss = 0
        mse_batches, psnr_batches, ssim_batches, perceptual_batches = [], [], [], []
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch.to(device)

                reconstruction, quantization_loss = model(images=images)

                # get the first sample from the first validation batch for visualization
                # purposes
                if val_step == 1:
                    intermediary_images.append(reconstruction[:n_example_images, 0])
                    writer.add_image(tag=f'Validation/Input', img_tensor=torch.tensor(images[:n_example_images, 0], dtype=torch.float32), global_step=i)
                    writer.add_image(tag=f'Validation/Output', img_tensor=torch.tensor(reconstruction[:n_example_images, 0], dtype=torch.float32), global_step=i)

                recons_loss = l1_loss(reconstruction.float(), images.float())

                val_loss += recons_loss.item()

                # Compute PSNR, SSIM, MSE, and LPIPS between input image and reconstructed image
                mse_batch = mean_flat((reconstruction[:, :, 8:-8, :] - images[:, :, 8:-8, :]) ** 2)
                psnr_batch = PSNR(reconstruction[:, :, 8:-8, :], images[:, :, 8:-8, :])
                ssim_batch = SSIM(reconstruction[:, :, 8:-8, :], images[:, :, 8:-8, :])
                perceptual_batch = perceptual_loss(reconstruction[:, :, 8:-8, :].float(), images[:, :, 8:-8, :].float())

                mse_batches.append(mse_batch.mean().cpu())
                psnr_batches.append(psnr_batch.cpu())
                ssim_batches.append(ssim_batch.cpu())
                perceptual_batches.append(perceptual_batch.cpu())

                # output_3_channels = reconstruction[:, :, 8:-8, :].repeat(1, 3, 1, 1)  # (Batch, Channels, Height, Width)
                # target_3_channels = images[:, :, 8:-8, :].repeat(1, 3, 1, 1)  # (Batch, Channels, Height, Width)
                # lpips_batch = LPIPS(output_3_channels, target_3_channels)
                # lpips_batches.append(lpips_batch.cpu())

        val_loss /= val_step
        val_recon_epoch_loss_list.append(val_loss)
        writer.add_scalar('Loss/val_loss', val_loss, epoch+1)

        psnr_batches = np.asarray(psnr_batches, dtype=np.float32)
        ssim_batches = np.asarray(ssim_batches, dtype=np.float32)
        mse_batches = np.asarray(mse_batches, dtype=np.float32)
        perceptual_batches = np.asarray(perceptual_batches, dtype=np.float32)

        # Calculate averages
        avg_psnr, std_psnr = np.mean(psnr_batches), np.std(psnr_batches)
        avg_ssim, std_ssim = np.mean(ssim_batches), np.std(ssim_batches)
        avg_mse, std_mse = np.mean(mse_batches), np.std(mse_batches)
        avg_perceptual, std_perceptual = np.mean(perceptual_batches), np.std(perceptual_batches)

        # Log average metrics to TensorBoard
        metrics_summary = {
            "PSNR": avg_psnr,
            "SSIM": avg_ssim,
            "MSE": avg_mse,
            "PERC_LOSS": avg_perceptual,
        }

        for metric_name, value in metrics_summary.items():
            writer.add_scalar(f"Validation_metrics/{metric_name}", value.item(), epoch + 1)
        print(
            f"Validation metrics, epoch {epoch + 1}: PSNR: {avg_psnr.item():.5f} ± {std_psnr.item():.5f} | SSIM: {avg_ssim.item():.5f} ± {std_ssim.item():.5f} | MSE: {avg_mse.item():.5f} ± {std_mse.item():.5f} | PERC_LOSS: {avg_perceptual.item():.5f} ± {std_perceptual.item():.5f}")

    # Save model checkpoint every 50 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"vqgan_epoch_{epoch+1}.pth")

        torch.save(model.state_dict(), checkpoint_path)  # Save only model weights

        print(f"Checkpoint saved at {checkpoint_path}")

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")
# %% [markdown]
# ### Cleanup data directory
#
# Remove directory if a temporary was used.

# %%
if directory is None:
    shutil.rmtree(root_dir)


# %% [markdown]
# ### Learning curves

"""
# %%
plt.style.use("seaborn-v0_8")
plt.title("Learning Curves", fontsize=20)
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_recon_loss_list, color="C0", linewidth=2.0, label="Train")
plt.plot(
    np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
    val_recon_epoch_loss_list,
    color="C1",
    linewidth=2.0,
    label="Validation",
)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.show()

# %%
plt.title("Adversarial Training Curves", fontsize=20)
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_gen_loss_list, color="C0", linewidth=2.0, label="Generator")
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_disc_loss_list, color="C1", linewidth=2.0, label="Discriminator")
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.show()

# %% [markdown]
# ### Checking reconstructions

# %%
# Plot first 4 evaluations
val_samples = np.linspace(val_interval, n_epochs, int(n_epochs / val_interval))
fig, ax = plt.subplots(nrows=4, ncols=1, sharey=True)
for image_n in range(4):
    reconstructions = torch.reshape(intermediary_images[image_n], (64 * n_example_images, 64)).T
    ax[image_n].imshow(reconstructions.cpu(), cmap="gray")
    ax[image_n].set_xticks([])
    ax[image_n].set_yticks([])
    ax[image_n].set_ylabel(f"Epoch {val_samples[image_n]:.0f}")


# %%
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(images[0, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
ax[0].axis("off")
ax[0].title.set_text("Inputted Image")
ax[1].imshow(reconstruction[0, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
ax[1].axis("off")
ax[1].title.set_text("Reconstruction")
plt.show()
"""
