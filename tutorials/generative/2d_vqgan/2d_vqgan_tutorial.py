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
from torchmetrics.image import PeakSignalNoiseRatio
from PIL import Image
import cv2
import argparse
from torchvision.utils import make_grid
print_config()


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


def create_grid(output_images, input_images, target_images):
    assert output_images.shape == input_images.shape == target_images.shape
    # Stack horizontally: [input | target | output]
    images = torch.stack([input_images, target_images, output_images], dim=0)
    grid = make_grid(images, nrow=3, normalize=False, value_range=None, padding=0)
    # writer.add_image(f'{tag_prefix}/sample_{b}', grid, iteration)
    return grid


def save_difference_maps_diffusion_paper(input_image, target_image, output_image, epoch, batch, histogram, writer, phase, folder):
    assert target_image.shape == output_image.shape == input_image.shape
    os.makedirs(os.path.join(folder, 'Difference_Maps'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'Histograms'), exist_ok=True)
    for i in range(target_image.shape[0]):
        # for each image in the batch
        # we remove the channel dimension and transform them to ndarray
        target = np.asarray(target_image[i, 0, ...], dtype=np.float32)
        output = np.asarray(output_image[i, 0, ...], dtype=np.float32)
        input = np.asarray(input_image[i, 0, ...], dtype=np.float32)

        # Compute the Difference Map
        difference_map = output - target

        vmin = -1
        vmax = 1
        fig = plt.figure()
        plt.imshow(difference_map, cmap='seismic', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title('Difference Map (Output-Target)')
        # plt.tight_layout()
        # Convert the Matplotlib figure to a NumPy array
        # plt_image = _plot_to_image(fig)
        # Add it to tensorboard
        # writer.add_image(f'{phase}_Difference_Maps/Sample_{batch}_{i+1}', plt_image, epoch)
        plt.savefig(f'{folder}/Difference_Maps/Sample_{batch}_{i + 1}.png')
        plt.close(fig)

        if histogram:
            # Compute the histograms of the pixel intensity distribution of target and model output
            hist_ground_truth = cv2.calcHist([target*255], [0], None, [256], [0, 256])
            hist_output = cv2.calcHist([output*255], [0], None, [256], [0, 256])
            hist_input = cv2.calcHist([input*255], [0], None, [256], [0, 256])

            hist_ground_truth /= hist_ground_truth.sum()
            hist_output /= hist_output.sum()
            hist_input /= hist_input.sum()

            fig = plt.figure(figsize=(10, 6))
            plt.plot(hist_ground_truth, color='blue', label='Ground Truth')
            plt.plot(hist_output, color='red', label='Output')
            plt.plot(hist_input, color='green', label='Input')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Normalized Frequency')
            plt.legend()
            plt.tight_layout()

            # plt_image = _plot_to_image(fig)
            # Add the figure on tensorboard
            # writer.add_image(f'{phase}_Histograms/Sample_{batch}_{i+1}', plt_image, epoch)
            plt.savefig(f'{folder}/Histograms/Sample_{batch}_{i + 1}.png')
            plt.close(fig)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Train VQGAN model")
    parser.add_argument("--num_embeddings", type=int, required=True, help="Number of embedding vectors")
    parser.add_argument("--embedding_dimension", type=int, required=True, help="Embedding dimension of each vector")
    parser.add_argument("--pixel_range", type=int, default=0, help="Pixel range of each image, 0 means [0,1], -1 means [-1,1]")
    # parser.add_argument("--iteration", type=int, default=0, help="Number of current epoch from which to sample")
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


# %%
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

set_determinism(1927)

args = parse_args_and_config()

train = np.load('/home/simone.sarrocco/thesis/project/data/train_set_patient_split.npz')['images']
val = np.load('/home/simone.sarrocco/thesis/project/data/val_set_patient_split.npz')['images']
test = np.load('/home/simone.sarrocco/thesis/project/data/test_set_patient_split.npz')['images']

# CURRENT SPLIT: 75% TRAIN (990 pairs), 12% VALIDATION (160 pairs), 13% TEST (170 pairs)
train_dataset = OCTDataset(train, resize=args.resize, pixel_range=args.pixel_range, clip=args.clip,
                           gaussian_noise=args.gaussian_noise, blur=args.blur, transform=args.transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataset = OCTDataset(val, resize=args.resize, pixel_range=args.pixel_range, clip=args.clip,
                         gaussian_noise=args.gaussian_noise, blur=args.blur, transform=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

test_dataset = OCTDataset(test, resize=args.resize, pixel_range=args.pixel_range, clip=args.clip,
                         gaussian_noise=args.gaussian_noise, blur=args.blur, transform=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print('Len train_dataset:', len(train_dataset))
print('Len train_dataloader:', len(train_loader))

print('Len val_dataset:', len(val_dataset))
print('Len val_dataloader:', len(val_loader))

print('Len test_dataset:', len(test_dataset))
print('Len test_dataloader:', len(test_loader))


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

discriminator = PatchDiscriminator(spatial_dims=2, in_channels=2, num_layers_d=3, num_channels=64)
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
    for step, (input_image, target_image) in progress_bar:
        art10 = input_image.to(device)
        pseudoart100 = target_image.to(device)

        optimizer_g.zero_grad(set_to_none=True)

        # Generator part
        reconstruction, quantization_loss = model(images=art10)

        i += 1
        concatenation_input_recon = torch.cat((art10, reconstruction), dim=1)
        # logits_fake = discriminator(reconstruction.contiguous().float())[-1]
        logits_fake = discriminator(concatenation_input_recon.contiguous().float())[-1]

        recons_loss = l1_loss(reconstruction.float(), pseudoart100.float())
        p_loss = perceptual_loss(reconstruction.float(), pseudoart100.float())
        generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
        loss_g = recons_loss + quantization_loss + perceptual_weight * p_loss + adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()

        # Discriminator part
        optimizer_d.zero_grad(set_to_none=True)

        #  logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
        logits_fake = discriminator(concatenation_input_recon.contiguous().detach())[-1]
        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        concatenation_input_target = torch.cat((art10, pseudoart100), dim=1)
        # logits_real = discriminator(images.contiguous().detach())[-1]
        logits_real = discriminator(concatenation_input_target.contiguous().detach())[-1]
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
                                     img_tensor=(art10[:n_example_images, 0, 8:-8, :]+1)/2,
                                     global_step=i)
                    writer.add_image(tag=f'Training/Target',
                                     img_tensor=(pseudoart100[:n_example_images, 0, 8:-8, :] + 1) / 2,
                                     global_step=i)
                    writer.add_image(tag=f'Training/Output', img_tensor=(reconstruction[:n_example_images, 0, 8:-8, :]+1)/2,
                                     global_step=i)
                elif args.pixel_range == 0:
                    writer.add_image(tag=f'Training/Input',
                                     # img_tensor=images[:n_example_images, 0, 8:-8, :],
                                     img_tensor=art10[:n_example_images, 0, 8:-8, :],
                                     global_step=i)
                    writer.add_image(tag=f'Training/Target',
                                     img_tensor=(pseudoart100[:n_example_images, 0, 8:-8, :] + 1) / 2,
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
        for val_step, (input_image, target_image) in enumerate(val_loader, start=1):
            art10 = input_image.to(device)
            pseudoart100 = target_image.to(device)

            # reconstruction, quantization_loss = model(images=images)
            reconstruction, quantization_loss = model(images=art10)

            # get the first sample from the first validation batch for visualization
            # purposes
            if val_step == 1:
                if args.pixel_range == -1:
                    intermediary_images.append((reconstruction[:n_example_images, 0]+1)/2)
                    # writer.add_image(tag=f'Validation/Input', img_tensor=images[:n_example_images, 0], global_step=i)
                    writer.add_image(tag=f'Validation/Input', img_tensor=(art10[:n_example_images, 0, 8:-8, :]+1)/2, global_step=i)
                    writer.add_image(tag=f'Validation/Target', img_tensor=(pseudoart100[:n_example_images, 0, 8:-8, :]+1)/2, global_step=i)
                    writer.add_image(tag=f'Validation/Output', img_tensor=(reconstruction[:n_example_images, 0, 8:-8, :]+1)/2,
                                     global_step=i)
                elif args.pixel_range == 0:
                    intermediary_images.append(reconstruction[:n_example_images, 0])
                    # writer.add_image(tag=f'Validation/Input', img_tensor=images[:n_example_images, 0], global_step=i)
                    writer.add_image(tag=f'Validation/Input', img_tensor=art10[:n_example_images, 0, 8:-8, :], global_step=i)
                    writer.add_image(tag=f'Validation/Target', img_tensor=pseudoart100[:n_example_images, 0, 8:-8, :], global_step=i)
                    writer.add_image(tag=f'Validation/Output', img_tensor=reconstruction[:n_example_images, 0, 8:-8, :],
                                     global_step=i)

            recons_loss = l1_loss(reconstruction.float(), pseudoart100.float())

            val_loss += recons_loss.item()

            # Compute PSNR, SSIM, and MSE between input and reconstructed image
            mse_batch = mean_flat((reconstruction[:, :, 8:-8, :] - pseudoart100[:, :, 8:-8, :]) ** 2)
            psnr_batch = PSNR(reconstruction[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])
            ssim_batch = SSIM._compute_metric(reconstruction[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])
            # perceptual_batch = perceptual_loss(reconstruction[:, :, 8:-8, :].float(), pseudoart100[:, :, 8:-8, :].float())

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
ckpt_path = f"/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/2d_vqgan/{args.model_name}/checkpoints/vqgan_best_checkpoint.ckpt"
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

PSNR = PeakSignalNoiseRatio().to(device)
# SSIM = StructuralSimilarityIndexMeasure().to(device)
SSIM = SSIMMetric(spatial_dims=2, reduction='mean_batch')
# LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
n_example_images = 1
best_epoch = checkpoint["epoch"]

outputs_dir = f"/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/2d_vqgan/{args.model_name}/testing"
os.makedirs(outputs_dir, exist_ok=True)

# Testing loop
with torch.no_grad():
    mse_batches, psnr_batches, ssim_batches = [], [], []
    for test_step, (input_image, target_image) in enumerate(test_loader):
        art10 = input_image.to(device)
        pseudoart100 = target_image.to(device)
        # reconstruction, quantization_loss = model(images=images)
        reconstruction, quantization_loss = model(images=art10)

        # get the first sample from the first validation batch for visualization
        # purposes
        if test_step == 1:
            if args.pixel_range == -1:
                # writer.add_image(tag=f'Validation/Input', img_tensor=images[:n_example_images, 0], global_step=i)
                writer.add_image(tag=f'Testing/Input', img_tensor=(art10[:n_example_images, 0, 8:-8, :] + 1) / 2,
                                 global_step=best_epoch)
                writer.add_image(tag=f'Testing/Target', img_tensor=(pseudoart100[:n_example_images, 0, 8:-8, :] + 1) / 2,
                                 global_step=best_epoch)
                writer.add_image(tag=f'Testing/Output',
                                 img_tensor=(reconstruction[:n_example_images, 0, 8:-8, :] + 1) / 2,
                                 global_step=best_epoch)
            elif args.pixel_range == 0:
                # writer.add_image(tag=f'Validation/Input', img_tensor=images[:n_example_images, 0], global_step=i)
                writer.add_image(tag=f'Testing/Input', img_tensor=art10[:n_example_images, 0, 8:-8, :],
                                 global_step=best_epoch)
                writer.add_image(tag=f'Testing/Target', img_tensor=pseudoart100[:n_example_images, 0, 8:-8, :],
                                 global_step=best_epoch)
                writer.add_image(tag=f'Testing/Output', img_tensor=reconstruction[:n_example_images, 0, 8:-8, :],
                                 global_step=best_epoch)

        for i in range(art10.shape[0]):
            # one grid for each image in each batch
            grid = create_grid(
                reconstruction[i, :, 8:-8, :],
                art10[i, :, 8:-8, :],
                pseudoart100[i, :, 8:-8, :]
            )
            grid_np = grid.permute(1, 2, 0).numpy()
            grid_np = (grid_np * 255).astype(np.uint8)
            final_grid = Image.fromarray(grid_np)
            final_grid.save(f'{outputs_dir}/Grids_Input_Target_Output/Sample_{test_step+1}_{i + 1}.png')
            # cv2.imwrite(filename=f'{path}/Grids_Input_Target_Output/Sample_{batch_idx + 1}_{i + 1}.png', img=grid)
            # self.writer.add_image(f'{phase}/Sample_{batch_idx + 1}_{i + 1}', grid,
            #                      self.step + self.resume_step)

        # We compute and save the difference map between output and target (in the range [0,1])
        save_difference_maps_diffusion_paper(
            art10, pseudoart100, reconstruction,
            best_epoch, test_step + 1,
            histogram=True,
            writer=writer,
            phase='Testing',
            folder=outputs_dir,
        )

        # Compute PSNR, SSIM, and MSE between input and reconstructed image
        mse_batch = mean_flat((reconstruction[:, :, 8:-8, :] - pseudoart100[:, :, 8:-8, :]) ** 2)
        psnr_batch = PSNR(reconstruction[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])
        ssim_batch = SSIM._compute_metric(reconstruction[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])
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
        writer.add_scalar(f"Testing_metrics/{metric_name}", value.item(), best_epoch)
    print(
        f"Testing metrics, epoch {best_epoch}: PSNR: {avg_psnr.item():.5f} ± {std_psnr.item():.5f} | SSIM: {avg_ssim.item():.5f} ± {std_ssim.item():.5f} | MSE: {avg_mse.item():.5f} ± {std_mse.item():.5f}")
