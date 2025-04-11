import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import importlib
from dataset import OCTDataset
from generative.metrics.ssim import SSIMMetric
from torchmetrics.image import PeakSignalNoiseRatio
# Import the necessary modules
from model import Encoder, Decoder
from quantize import VectorQuantizer2 as VectorQuantizer
from quantize import GumbelQuantize
from lpips import LPIPS
import argparse
from utils import dict2namespace, get_runner, namespace2dict
import yaml


# Set device
device = torch.device("cuda")
print(f"Using device: {device}")

PSNR = PeakSignalNoiseRatio().to(device)
# SSIM = StructuralSimilarityIndexMeasure().to(device)
SSIM = SSIMMetric(spatial_dims=2)
# LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    if config.__contains__('params'):
        return get_obj_from_str(config["target"])(**vars(config['params']))
    else:
        return get_obj_from_str(config["target"])()


class VQModel(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**vars(ddconfig))
        self.decoder = Decoder(**vars(ddconfig))
        self.loss = instantiate_from_config(vars(lossconfig))
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig.z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig.z_channels, 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        # sd = torch.load(path, map_location="cpu")["state_dict"]
        checkpoint = torch.load(path, map_location="cpu")
        sd = checkpoint["model_state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch):
        x = batch
        # if len(x.shape) == 3:
        #    x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()

    def get_last_layer(self):
        return self.decoder.conv_out.weight


def train_vqgan(config, train_loader, val_loader, device, output_dir, num_epochs=100,
                save_interval=10, sample_interval=10):
    """
    Train the VQGAN model with pure PyTorch.

    Args:
        config: Model configuration
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on
        output_dir: Directory to save outputs
        num_epochs: Number of epochs to train
        save_interval: Interval for saving checkpoints
        sample_interval: Interval for sampling and computing metrics
    """
    # Create directories
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    sample_dir = os.path.join(output_dir, "samples")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

    # Initialize model
    model = VQModel(
        ddconfig=config.model.VQGAN.params.ddconfig,
        lossconfig=config.model.VQGAN.params.lossconfig,
        n_embed=config.model.VQGAN.params.n_embed,
        embed_dim=config.model.VQGAN.params.embed_dim,
    ).to(device)

    # Initialize optimizers
    opt_ae = torch.optim.Adam(
        list(model.encoder.parameters()) +
        list(model.decoder.parameters()) +
        list(model.quantize.parameters()) +
        list(model.quant_conv.parameters()) +
        list(model.post_quant_conv.parameters()),
        lr=1e-4,
        betas=(0.5, 0.9)
    )

    opt_disc = torch.optim.Adam(
        model.loss.discriminator.parameters(),
        lr=1e-4,
        betas=(0.5, 0.9)
    )

    # Initialize perceptual loss for evaluation
    perceptual_loss = LPIPS().eval().to(device)

    # L1 loss for reconstruction
    l1_loss = torch.nn.L1Loss()

    # Track best validation loss
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_ae = 0.0
        train_loss_disc = 0.0
        train_samples = 0

        # Training epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch.to(device)

            # Prepare input
            x = model.get_input(images)

            # Encoder and quantization
            xrec, qloss = model(x)

            # Train autoencoder (Generator)
            opt_ae.zero_grad()
            aeloss, log_dict_ae = model.loss(
                qloss, x, xrec, 0, batch_idx + (epoch - 1) * len(train_loader),
                last_layer=model.get_last_layer(), split="train"
            )
            aeloss.backward()
            opt_ae.step()

            # Train discriminator
            opt_disc.zero_grad()
            discloss, log_dict_disc = model.loss(
                qloss, x, xrec, 1, batch_idx + (epoch - 1) * len(train_loader),
                last_layer=model.get_last_layer(), split="train"
            )
            discloss.backward()
            opt_disc.step()

            # Update metrics
            train_loss_ae += aeloss.item() * x.size(0)
            train_loss_disc += discloss.item() * x.size(0)
            train_samples += x.size(0)

            # Update progress bar
            pbar.set_postfix({
                "ae_loss": aeloss.item(),
                "disc_loss": discloss.item()
            })

        # Calculate average training losses
        train_loss_ae /= train_samples
        train_loss_disc /= train_samples

        # Log training losses
        writer.add_scalar("Train/AE_Loss", train_loss_ae, epoch)
        writer.add_scalar("Train/Disc_Loss", train_loss_disc, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0

        # Initialize metric lists
        mse_batches = []
        psnr_batches = []
        ssim_batches = []
        perceptual_batches = []

        # Sample some images for visualization (only every sample_interval epochs)
        intermediary_images = []
        n_example_images = 4  # Number of example images to visualize

        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]")
        with torch.no_grad():
            for val_step, batch in enumerate(pbar, start=1):
                images = batch.to(device)
                x = model.get_input(images)

                # Forward pass
                reconstruction, quantization_loss = model(x)

                # Save example images for visualization
                if epoch % sample_interval == 0 and val_step == 1:
                    # Get the first few samples from validation set
                    intermediary_images.append(reconstruction[:n_example_images])
                    writer.add_images(tag=f'Validation/Input', img_tensor=x[:n_example_images], global_step=epoch)
                    writer.add_images(tag=f'Validation/Output', img_tensor=reconstruction[:n_example_images],
                                      global_step=epoch)

                # Compute reconstruction loss
                recons_loss = l1_loss(reconstruction.float(), x.float())
                val_loss += recons_loss.item() * x.size(0)
                val_samples += x.size(0)

                # Compute metrics (every sample_interval epochs)
                if epoch % sample_interval == 0:
                    # Compute on cropped images (8 pixels from top/bottom) to avoid boundary effects
                    mse_batch = mean_flat((reconstruction[:, :, 8:-8, :] - x[:, :, 8:-8, :]) ** 2)
                    psnr_batch = PSNR(reconstruction[:, :, 8:-8, :], x[:, :, 8:-8, :])
                    ssim_batch = SSIM._compute_metric(reconstruction[:, :, 8:-8, :], x[:, :, 8:-8, :])
                    perceptual_batch = perceptual_loss(reconstruction[:, :, 8:-8, :].float(), x[:, :, 8:-8, :].float())

                    mse_batches.append(mse_batch.mean().cpu())
                    psnr_batches.append(psnr_batch.cpu())
                    ssim_batches.append(ssim_batch.cpu())
                    perceptual_batches.append(perceptual_batch.cpu())

        # Calculate average validation loss
        val_loss /= val_samples
        writer.add_scalar("Validation/Loss", val_loss, epoch)

        # Log metrics (every sample_interval epochs)
        if epoch % sample_interval == 0:
            mse_avg = torch.stack(mse_batches).mean().item()
            psnr_avg = torch.stack(psnr_batches).mean().item()
            ssim_avg = torch.stack(ssim_batches).mean().item()
            perceptual_avg = torch.stack(perceptual_batches).mean().item()

            writer.add_scalar("Metrics/MSE", mse_avg, epoch)
            writer.add_scalar("Metrics/PSNR", psnr_avg, epoch)
            writer.add_scalar("Metrics/SSIM", ssim_avg, epoch)
            writer.add_scalar("Metrics/LPIPS", perceptual_avg, epoch)

            print(
                f"Epoch {epoch}: Val Loss: {val_loss:.6f}, MSE: {mse_avg:.6f}, PSNR: {psnr_avg:.6f}, SSIM: {ssim_avg:.6f}, LPIPS: {perceptual_avg:.6f}")
        else:
            print(f"Epoch {epoch}: Val Loss: {val_loss:.6f}")

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_ae_state_dict': opt_ae.state_dict(),
                'optimizer_disc_state_dict': opt_disc.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(checkpoint_dir, "vqgan_best.pt"))
            print(f"Saved best model with validation loss: {val_loss:.6f}")

        # Regular checkpoint saving
        if epoch % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_ae_state_dict': opt_ae.state_dict(),
                'optimizer_disc_state_dict': opt_disc.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(checkpoint_dir, f"vqgan_epoch_{epoch}.pt"))

    writer.close()
    print("Training completed!")


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Train VQGAN model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--save_interval", type=int, default=50, help="Interval for saving checkpoints")
    parser.add_argument("--sample_interval", type=int, default=10, help="Interval for sampling and computing metrics")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args

    dict_config = namespace2dict(namespace_config)

    return namespace_config, dict_config


def main():
    nconfig, dconfig = parse_args_and_config()
    args = nconfig.args

    # Set up data loaders
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

    val_data = OCTDataset(val_images)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=64)
    print(f'Shape of validation set: {val_images.shape}')

    # Train model
    train_vqgan(
        config=nconfig,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval
    )


if __name__ == "__main__":
    main()
