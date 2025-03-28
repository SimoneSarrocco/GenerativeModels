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

# # Vector Quantized Variational Autoencoders with MedNIST Dataset
#
# This tutorial illustrates how to use MONAI for training a Vector Quantized Variational Autoencoder (VQVAE)[1] on 2D images.
#
# Here, we will train our VQVAE model to be able to reconstruct the input images.  We will work with the MedNIST dataset available on MONAI
# (https://docs.monai.io/en/stable/apps.html#monai.apps.MedNISTDataset). In order to train faster, we will select just one of the available classes ("HeadCT"), resulting in a training set with 7999 2D images.
#
# The VQVAE can also be used as a generative model if an autoregressor model (e.g., PixelCNN, Decoder Transformer) is trained on the discrete latent representations of the VQVAE bottleneck. This falls outside of the scope of this tutorial.
#
# [1] - Oord et al. "Neural Discrete Representation Learning" https://arxiv.org/abs/1711.00937
#
#
# ### Setup environment

# !python -c "import monai" || pip install -q "monai-weekly[tqdm]"
# !python -c "import matplotlib" || pip install -q matplotlib
# %matplotlib inline


# ### Setup imports

# +
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
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from torch.nn import L1Loss
from tqdm import tqdm
from dataset import OCTDataset
from generative.networks.nets import VQVAE
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from generative.metrics.ssim import SSIMMetric


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

    checkpoint_path = os.path.join(save_dir, f"vqvae_epoch_{epoch}.ckpt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

print_config()
# -

# for reproducibility purposes set a seed
set_determinism(1927)

# ### Setup a data directory and download dataset

# Specify a `MONAI_DATA_DIRECTORY` variable, where the data will be downloaded. If not
# specified a temporary directory will be used.

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

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

train_data = OCTDataset(train_images, transform=True)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=64)
print(f'Shape of training set: {train_images.shape}')
# train_datalist = [{"image": train[i, -1:, ...]} for i in range(len(train))]

# %% [markdown]
# Here we use transforms to augment the training dataset:
#
# 1. `LoadImaged` loads the hands images from files.
# 1. `EnsureChannelFirstd` ensures the original data to construct "channel first" shape.
# 1. `ScaleIntensityRanged` extracts intensity range [0, 255] and scales to [0, 1].
# 1. `RandAffined` efficiently performs rotate, scale, shear, translate, etc. together based on PyTorch affine transform.

# %%
val_data = OCTDataset(val_images)
# val_datalist = [{"image": val[i, -1:, ...]} for i in range(len(val))]
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=64)
print(f'Shape of validation set: {val_images.shape}')

# ### Define network, optimizer and losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
model = VQVAE(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 128, 256, 256, 512, 512),
    num_res_channels=512,
    num_res_layers=2,
    downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
    upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
    num_embeddings=16384,
    embedding_dim=8,
)
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
l1_loss = L1Loss()

run_name = 'reconstruction_new_embed_dim_8_longer_architecture'

tensorboard_dir = f'/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/2d_vqvae/{run_name}/tensorboard_log'
writer = SummaryWriter(log_dir=tensorboard_dir)
# Define the directory to save checkpoints
checkpoint_dir = f"/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/2d_vqvae/{run_name}/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# ### Model training
# Here, we are training our model for 100 epochs (training time: ~60 minutes).

# +
n_epochs = 100
val_interval = 1
epoch_recon_loss_list = []
epoch_quant_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []
n_example_images = 1

i = 0
PSNR = PeakSignalNoiseRatio().to(device)
# SSIM = StructuralSimilarityIndexMeasure().to(device)
SSIM = SSIMMetric(spatial_dims=2)
# LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
total_start = time.time()
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    mse_batches, psnr_batches, ssim_batches, perceptual_batches = [], [], [], []
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        # model outputs reconstruction and the quantization error
        reconstruction, quantization_loss = model(images=images)
        i += 1
        recons_loss = l1_loss(reconstruction.float(), images.float())

        loss = recons_loss + quantization_loss

        loss.backward()
        optimizer.step()

        epoch_loss += recons_loss.item()

        progress_bar.set_postfix(
            {"recons_loss": epoch_loss / (step + 1), "quantization_loss": quantization_loss.item() / (step + 1)}
        )
        with torch.no_grad():
            writer.add_scalar('Loss/recons_loss', recons_loss, i)

            # Every epoch visualize input image and corresponding reconstruction on tensorboard
            if i % 1980 == 0:
                writer.add_image(tag=f'Training/Input',
                                 # img_tensor=images[:n_example_images, 0, 8:-8, :],
                                 img_tensor=images[:n_example_images, 0, 8:-8, :],
                                 global_step=i)
                writer.add_image(tag=f'Training/Output', img_tensor=reconstruction[:n_example_images, 0, 8:-8, :],
                                 global_step=i)

    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_quant_loss_list.append(quantization_loss.item() / (step + 1))

    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch.to(device)

                reconstruction, quantization_loss = model(images=images)

                # get the first sample from the first validation batch for
                # visualizing how the training evolves
                if val_step == 1:
                    intermediary_images.append(reconstruction[:n_example_images, 0])
                    # writer.add_image(tag=f'Validation/Input', img_tensor=images[:n_example_images, 0], global_step=i)
                    writer.add_image(tag=f'Validation/Input', img_tensor=images[:n_example_images, 0],
                                     global_step=i)
                    writer.add_image(tag=f'Validation/Output', img_tensor=reconstruction[:n_example_images, 0],
                                     global_step=i)

                recons_loss = l1_loss(reconstruction.float(), images.float())

                val_loss += recons_loss.item()

                # Compute PSNR, SSIM, MSE, and LPIPS between input image and reconstructed image
                # mse_batch = mean_flat((reconstruction[:, :, 8:-8, :] - images[:, :, 8:-8, :]) ** 2)
                mse_batch = mean_flat((reconstruction[:, :, 8:-8, :] - images[:, :, 8:-8, :]) ** 2)
                # psnr_batch = PSNR(reconstruction[:, :, 8:-8, :], images[:, :, 8:-8, :])
                psnr_batch = PSNR(reconstruction[:, :, 8:-8, :], images[:, :, 8:-8, :])
                # ssim_batch = SSIM._compute_metric(reconstruction[:, :, 8:-8, :], images[:, :, 8:-8, :])
                ssim_batch = SSIM._compute_metric(reconstruction[:, :, 8:-8, :], images[:, :, 8:-8, :])
                # perceptual_batch = perceptual_loss(reconstruction[:, :, 8:-8, :].float(), images[:, :, 8:-8, :].float())
                # perceptual_batch = perceptual_loss(reconstruction[:, :, 8:-8, :].float(), images[:, :, 8:-8, :].float())

                mse_batches.append(mse_batch.mean().cpu())
                psnr_batches.append(psnr_batch.cpu())
                ssim_batches.append(ssim_batch.cpu())
                # perceptual_batches.append(perceptual_batch.cpu())

        val_loss /= val_step
        val_recon_epoch_loss_list.append(val_loss)

        writer.add_scalar('Loss/val_loss', val_loss, epoch + 1)

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

        # Save model checkpoint every 50 epochs
    if (epoch + 1) % 10 == 0:
        save_checkpoint(model, epoch + 1, save_dir=checkpoint_dir)

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")
# -

# ### Learning curves

plt.style.use("ggplot")
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

# ###  Plotting  evolution of reconstructed images

"""# Plot every evaluation as a new line and example as columns
val_samples = np.linspace(val_interval, n_epochs, int(n_epochs / val_interval))
fig, ax = plt.subplots(nrows=len(val_samples), ncols=1, sharey=True)
fig.set_size_inches(18.5, 30.5)
for image_n in range(len(val_samples)):
    reconstructions = torch.reshape(intermediary_images[image_n], (64 * n_example_images, 64)).T
    ax[image_n].imshow(reconstructions.cpu(), cmap="gray")
    ax[image_n].set_xticks([])
    ax[image_n].set_yticks([])
    ax[image_n].set_ylabel(f"Epoch {val_samples[image_n]:.0f}")


# ### Plotting the reconstructions from final trained model

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(images[0, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
ax[0].axis("off")
ax[0].title.set_text("Inputted Image")
ax[1].imshow(reconstruction[0, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
ax[1].axis("off")
ax[1].title.set_text("Reconstruction")
plt.show()"""

# ### Cleanup data directory
#
# Remove directory if a temporary was used.

if directory is None:
    shutil.rmtree(root_dir)
