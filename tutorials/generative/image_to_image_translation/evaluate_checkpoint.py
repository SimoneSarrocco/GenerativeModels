#!/usr/bin/env python
# DDPM Checkpoint Evaluation Script

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Import necessary MONAI components
from monai.utils import set_determinism
from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddpm import DDPMScheduler
from generative.losses import PerceptualLoss
from generative.metrics.ssim import SSIMMetric
from torchmetrics.image import PeakSignalNoiseRatio

# Assume the OCTDataset class is in the same directory
from dataset import OCTDataset


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def load_model(checkpoint_path, device):
    """
    Load the model from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to

    Returns:
        model: The loaded DiffusionModelUNet model
    """
    print(f"Loading model from {checkpoint_path}")

    # Initialize the model with the same architecture used during training
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=2,
        out_channels=1,
        num_channels=(128, 128, 256, 256, 512, 512),
        attention_levels=(False, False, False, False, False, True),
        num_res_blocks=2,
        num_head_channels=64,
        with_conditioning=False,
    ).to(device)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # If the checkpoint contains only the model state
    if isinstance(checkpoint, dict) and all(k.startswith("module.") or "." in k for k in checkpoint.keys()):
        model.load_state_dict(checkpoint)
    # If the checkpoint contains the full training state
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("Unrecognized checkpoint format")

    return model


def evaluate_model(model, val_loader, device, writer, epoch, output_dir=None, num_samples=None, save_images=False, phase=None):
    """
    Evaluate a DDPM model on the validation set.

    Args:
        model: The trained DiffusionModelUNet
        val_loader: DataLoader for validation data
        device: Device to run evaluation on
        output_dir: Directory to save outputs (optional)
        num_samples: Number of samples to evaluate (None for all)
        save_images: Whether to save generated images

    Returns:
        metrics_dict: Dictionary containing evaluation metrics
    """
    model.eval()

    # Initialize metric calculations
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    inferer = DiffusionInferer(scheduler)

    # Setup metrics
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = SSIMMetric(spatial_dims=2)
    perceptual_metric = PerceptualLoss(spatial_dims=2, device=device)

    # Create output directory if needed
    if save_images and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Lists to store batch metrics
    mse_values = []
    psnr_values = []
    ssim_values = []
    perceptual_values = []

    # Process validation samples
    start_time = time.time()

    # Limit samples if specified
    sample_count = 0
    val_loader_with_progress = tqdm(val_loader, desc="Evaluating")

    for step, (art10, pseudoart100) in enumerate(val_loader_with_progress):
        if num_samples is not None and sample_count >= num_samples:
            break

        art10 = art10.to(device)
        pseudoart100 = pseudoart100.to(device)

        # Start from random noise
        noise = torch.randn_like(pseudoart100).to(device)
        current_img = noise.clone()
        combined = torch.cat((art10, noise), dim=1)

        # Setup diffusion timesteps
        scheduler.set_timesteps(num_inference_steps=1000)
        progress_bar = tqdm(scheduler.timesteps, desc=f"Sampling {step + 1}", leave=False)

        # Sample loop - gradually denoise the image
        for t in progress_bar:
            with autocast("cuda"):
                with torch.no_grad():
                    model_output = model(combined, timesteps=torch.Tensor((t,)).to(device))
                    current_img, _ = scheduler.step(model_output, t, current_img)
                    combined = torch.cat((art10, current_img), dim=1)

        if step == 0:
            writer.add_image(f'{phase}/Input', art10.squeeze(0), epoch)
            writer.add_image(f'{phase}/Output', current_img.clamp(0., 1.).squeeze(0), epoch)
            writer.add_image(f'{phase}/Target', pseudoart100.squeeze(0), epoch)

        # Save generated images if requested
        if save_images and output_dir is not None:
            # Save input, output, and target images
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(art10[0, 0].cpu().numpy(), cmap='gray')
            axes[0].set_title('Input (ART10)')
            axes[0].axis('off')

            axes[1].imshow(current_img[0, 0].cpu().numpy(), cmap='gray')
            axes[1].set_title('Generated (pseudoART100)')
            axes[1].axis('off')

            axes[2].imshow(pseudoart100[0, 0].cpu().numpy(), cmap='gray')
            axes[2].set_title('Target (pseudoART100)')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{step + 1}.png'), dpi=300)
            plt.close(fig)

        # Compute metrics (excluding boundary pixels as in your training code)
        with torch.no_grad():
            mse_batch = mean_flat((current_img[:, :, 8:-8, :] - pseudoart100[:, :, 8:-8, :]) ** 2)
            psnr_batch = psnr_metric(current_img[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])
            ssim_batch = ssim_metric._compute_metric(current_img[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])
            # perceptual_batch = perceptual_metric(current_img[:, :, 8:-8, :], pseudoart100[:, :, 8:-8, :])

            mse_values.append(mse_batch.mean().cpu().numpy())
            psnr_values.append(psnr_batch.cpu().numpy())
            ssim_values.append(ssim_batch.cpu().numpy())
            # perceptual_values.append(perceptual_batch.cpu().numpy())

        sample_count += 1

    # Calculate aggregate metrics
    metrics = {
        'MSE': (np.mean(mse_values), np.std(mse_values)),
        'PSNR': (np.mean(psnr_values), np.std(psnr_values)),
        'SSIM': (np.mean(ssim_values), np.std(ssim_values)),
        # 'Perceptual': (np.mean(perceptual_values), np.std(perceptual_values))
    }

    elapsed_time = time.time() - start_time

    # Print results
    print(f"\nEvaluation Results for checkpoint {epoch} on {phase} set:")
    print(f"Evaluated on {sample_count} samples in {elapsed_time:.2f} seconds")
    for metric_name, (mean_val, std_val) in metrics.items():
        print(f"{metric_name}: {mean_val:.5f} ± {std_val:.5f}")
        writer.add_scalar(f"{phase}_metrics/{metric_name}", mean_val, epoch)

    # Save metrics to file if output directory is provided
    if output_dir is not None:
        metrics_file = os.path.join(output_dir, "evaluation_metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"Evaluation on {sample_count} samples\n")
            f.write(f"Time taken: {elapsed_time:.2f} seconds\n\n")
            for metric_name, (mean_val, std_val) in metrics.items():
                f.write(f"{metric_name}: {mean_val:.5f} ± {std_val:.5f}\n")

        # Save raw metric values for future analysis
        np.savez(
            os.path.join(output_dir, "raw_metrics.npz"),
            mse=np.array(mse_values),
            psnr=np.array(psnr_values),
            ssim=np.array(ssim_values),
            perceptual=np.array(perceptual_values)
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate DDPM model checkpoint on validation data")
    parser.add_argument("--checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--val_data", type=str, default="/home/simone.sarrocco/thesis/project/data/val_set_patient_split.npz", help="Path to validation data (.npz file)")
    parser.add_argument("--run_number", type=str, default="1st_run", help="Name of the model configuration")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--save_images", action="store_true", help="Save generated images")
    parser.add_argument("--seed", type=int, default=1927, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--epoch_number", type=int, default=0, help="Epoch number")
    parser.add_argument("--phase", type=str, default='Validation', help="Validation or Test")
    args = parser.parse_args()

    checkpoint = f'/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/image_to_image_translation/checkpoints/{args.run_number}/ddpm_oct_model_{args.epoch_number}.pt'

    output_dir = f'/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/image_to_image_translation/results/{args.run_number}/{args.phase}/{args.epoch_number}/output_samples'
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    writer = SummaryWriter(
        log_dir=f'/home/simone.sarrocco/thesis/project/models/diffusion_model/GenerativeModels/tutorials/generative/image_to_image_translation/logs/{args.run_number}/{args.epoch_number}')

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        print("Warning: CUDA not available, using CPU")

    # Set deterministic behavior
    set_determinism(args.seed)

    # Load validation data
    print(f"Loading {args.phase} data from {args.val_data}")
    val_data_np = np.load(args.val_data)['images']
    val_dataset = OCTDataset(val_data_np, transform=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"{args.phase} set size: {len(val_dataset)}")

    # Load model
    model = load_model(checkpoint, device)

    # Evaluate model
    metrics = evaluate_model(
        model,
        val_loader,
        device,
        writer,
        epoch=args.epoch_number,
        output_dir=output_dir,
        num_samples=args.num_samples,
        save_images=args.save_images,
        phase=args.phase,
    )

    print("Evaluation complete!")


if __name__ == "__main__":
    main()