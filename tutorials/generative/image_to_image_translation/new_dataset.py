from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import glob
import os
import torch
import torch.nn.functional as F

class OCTFolderDataset(Dataset):
    def __init__(self, root_folder, transform=None, num_inputs=10, padding=(0, 8, 0, 8)):
        """
        Args:
            root_folder (str): Path to ART10_21Bscans folder.
            transform (callable, optional): Transform to be applied to the images.
            num_inputs (int): Number of ART10 images to use per location (default=10).
            padding (tuple): Padding applied as (left, right, top, bottom).
        """
        self.root_folder = root_folder
        self.transform = transform if transform else transforms.ToTensor()
        self.num_inputs = num_inputs
        self.padding = padding
        self.samples = []

        # Scan each location folder (0000, 0001, ..., 0020)
        for location_folder in sorted(os.listdir(root_folder)):
            location_path = os.path.join(root_folder, location_folder)

            if not os.path.isdir(location_path):
                continue

            # Get all ART10 files in this location folder
            art10_files = sorted(glob.glob(os.path.join(location_path, "*.tiff")))

            if len(art10_files) < self.num_inputs:
                print(f"Warning: {location_folder} has only {len(art10_files)} images, skipping.")
                continue

            # Use ONLY the first N images for both input and target
            selected_files = art10_files[:self.num_inputs]

            self.samples.append({
                "location": location_folder,
                "input_files": selected_files,
                "target_files": selected_files
            })

        print(f"Found {len(self.samples)} valid locations with at least {self.num_inputs} images each.")

    def __len__(self):
        # Total = number of locations × 10 inputs per location
        return len(self.samples) * self.num_inputs

    def _apply_padding(self, tensor):
        """Apply symmetric padding (top/bottom only, as used in your code)."""
        if self.padding is not None:
            padding = transforms.Pad(self.padding, fill=0)
            tensor = padding(tensor)
        return tensor

    def __getitem__(self, idx):
        """
        Returns:
            art10_tensor: Single ART10 image (input) with padding.
            pseudo_target_tensor: Averaged tensor of the same 10 ART10 images with padding.
            meta: Dictionary with location ID and file paths.
        """
        # Determine which location folder this index belongs to
        location_idx = idx // self.num_inputs  # Which location
        image_idx = idx % self.num_inputs      # Which image within the 10

        sample = self.samples[location_idx]
        input_file = sample["input_files"][image_idx]

        # --- Load single ART10 input ---
        art10_image = Image.open(input_file).convert("L")     # Step 1: PIL
        art10_tensor = self.transform(art10_image)            # Step 2: Tensor
        art10_tensor = self._apply_padding(art10_tensor)      # Step 3: Padding

        # --- Load and average the same 10 images to create pseudo target ---
        target_tensors = []
        for tf in sample["target_files"]:
            img = Image.open(tf).convert("L")                 # Step 1: PIL
            t_tensor = self.transform(img)                    # Step 2: Tensor
            t_tensor = self._apply_padding(t_tensor)          # Step 3: Padding
            target_tensors.append(t_tensor)

        pseudo_target_tensor = torch.stack(target_tensors, dim=0).mean(dim=0)

        meta = {
            "location": sample["location"],
            "input_file": input_file,
            "target_files": sample["target_files"]
        }

        return art10_tensor, pseudo_target_tensor, meta

