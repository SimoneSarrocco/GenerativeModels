import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as F
from PIL import Image
import ants


class OCTDataset(Dataset):
    def __init__(self, images, resize=False, pixel_range=0, gaussian_noise=False, clip=False, blur=False,
                 transform=False):
        self.image_pairs = images
        self.pixel_range = pixel_range
        self.resize = resize
        self.gaussian_noise = gaussian_noise
        self.clip = clip
        self.blur = blur
        self.transform = transform

    def __getitem__(self, index):
        low_snr = self.image_pairs[index, 0, ...]  # int, range [0, 255], and shape [H, W]
        high_snr = self.image_pairs[index, 1, ...]  # int, range [0, 255], and shape [H, W]

        low_snr = np.asarray(low_snr, dtype=np.float32)
        high_snr = np.asarray(high_snr, dtype=np.float32)

        # clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(15, 15))
        # low_snr = clahe.apply(low_snr)
        # Transform the ndarrays to float32 tensors of shape [1, H, W]
        # low_snr = torch.tensor(low_snr, dtype=torch.float32).unsqueeze(0)
        # high_snr = torch.tensor(high_snr, dtype=torch.float32).unsqueeze(0)

        if self.clip:
            # Clipping
            low_snr = np.clip(low_snr, np.quantile(low_snr, 0.001), np.quantile(low_snr, 0.999))
            high_snr = np.clip(high_snr, np.quantile(high_snr, 0.001), np.quantile(high_snr, 0.999))

        # Gaussian noise
        if self.gaussian_noise:
            # low_snr = cv2.GaussianBlur(low_snr, (3, 3), 1)
            low_snr = (low_snr - np.mean(low_snr)) / np.std(low_snr)
            # high_snr = (high_snr - np.mean(high_snr)) / np.std(high_snr)
            gaussian_noise = np.random.normal(0, 0.1, low_snr.shape)
            low_snr = low_snr + gaussian_noise
            # low_snr = np.clip(low_snr, 0, 1)
            # low_snr = cv2.addWeighted(low_snr, 1.5, gaussianblurred, -0.5, 0)
            # clahe = cv2.createCLAHE(clipLimit=127.5, tileGridSize=(8, 8))
            # low_snr = clahe.apply(low_snr.astype(np.uint8))

        if self.blur:
            low_snr = cv2.GaussianBlur(low_snr, (5, 5), 1)

        # Transform the ndarrays to float32 tensors of shape [1, H, W]
        low_snr = torch.tensor(low_snr, dtype=torch.float32).unsqueeze(0)
        high_snr = torch.tensor(high_snr, dtype=torch.float32).unsqueeze(0)

        if self.resize:
            resize = transforms.Resize((248, 384))
            low_snr = resize(low_snr)
            high_snr = resize(high_snr)
            if self.pixel_range == 0:
                padding = transforms.Pad((0, 4, 0, 4), fill=0)
            elif self.pixel_range == -1:
                padding = transforms.Pad((0, 4, 0, 4), fill=-1)
        else:
            if self.pixel_range == 0:
                padding = transforms.Pad((0, 8, 0, 8), fill=0)
            elif self.pixel_range == -1:
                padding = transforms.Pad((0, 8, 0, 8), fill=-1)

        # Standardize them
        # low_snr = (low_snr - torch.mean(low_snr)) / torch.std(low_snr)
        # high_snr = (high_snr - torch.mean(high_snr)) / torch.std(high_snr)

        # Normalize to [0,1] each image separately by min-max
        low_snr = (low_snr - torch.min(low_snr)) / (
                torch.max(low_snr) - torch.min(low_snr))

        high_snr = (high_snr - torch.min(high_snr)) / (
                torch.max(high_snr) - torch.min(high_snr))

        if self.transform:
            trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomRotation(degrees=5),
                # transforms.ColorJitter(0.2, 0.2),
            ])
            concatenation = torch.stack((low_snr, high_snr), dim=0)
            transformed_images = trans(concatenation)
            low_snr = transformed_images[0]
            high_snr = transformed_images[1]

        """
        # Standardize them
        low_snr = (low_snr - torch.mean(low_snr)) / torch.std(low_snr)
        high_snr = (high_snr - torch.mean(high_snr)) / torch.std(high_snr)

        # Normalize them to [0,1]
        preprocessed_low_snr = (preprocessed_low_snr - torch.min(preprocessed_low_snr)) / (
                torch.max(preprocessed_low_snr) - torch.min(preprocessed_low_snr))

        preprocessed_high_snr = (preprocessed_high_snr - torch.min(preprocessed_high_snr)) / (
                torch.max(preprocessed_high_snr) - torch.min(preprocessed_high_snr))
        """
        if self.pixel_range == -1:
            # Linearly scale each image from [0,1] to [-1,1]
            # preprocessed_low_snr = 2 * preprocessed_low_snr - 1
            # preprocessed_high_snr = 2 * preprocessed_high_snr - 1
            normalization = transforms.Normalize(0.5, 0.5)
            low_snr = normalization(low_snr)
            high_snr = normalization(high_snr)

        # Pad each image to either [256, 384] or [496, 768] depending on whether we have resized the images or not
        low_snr = padding(low_snr)
        high_snr = padding(high_snr)

        return low_snr, high_snr

    def __len__(self):
        return len(self.image_pairs)
