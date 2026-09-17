"""
eee_dataset.py
--------------
EEE Domain-Specific Image Dataset Module:
Generates thermographic Infrared (IR) and Electroluminescence (EL) solar photovoltaic (PV)
panel images for automated fault detection in Electrical and Electronics Engineering (EEE).

Classes:
  0: Healthy_Panel
  1: Micro_Crack
  2: Hotspot_Fault
  3: Dust_Soiling
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image


class SolarPVFaultDataset(Dataset):
    """
    Synthetic physics-based thermographic dataset representing Solar Photovoltaic (PV) panel images.

    Parameters
    ----------
    num_samples : int
        Total number of images to generate across all classes.
    image_size : tuple of (int, int)
        Dimensions of panel images (H, W), default (28, 28).
    seed : int
        Random seed for reproducible dataset synthesis.
    transform : callable, optional
        Torchvision transform pipeline.
    """

    CLASS_NAMES = ["Healthy_Panel", "Micro_Crack", "Hotspot_Fault", "Dust_Soiling"]

    def __init__(self, num_samples: int = 7000, image_size: tuple = (28, 28), seed: int = 42, transform=None):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        self.rng = np.random.RandomState(seed)

        self.images, self.labels = self._generate_dataset()

    def _generate_panel_base(self) -> np.ndarray:
        """Generate clean PV cell grid matrix with busbars and cell boundaries."""
        h, w = self.image_size
        img = np.full((h, w), 0.35, dtype=np.float32)

        # Draw grid lines (busbars and cell borders)
        grid_step_y = max(4, h // 4)
        grid_step_x = max(4, w // 4)

        for y in range(0, h, grid_step_y):
            img[y, :] = 0.65
        for x in range(0, w, grid_step_x):
            img[:, x] = 0.60

        # Sensor background noise
        noise = self.rng.normal(0.0, 0.03, (h, w)).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)
        return img

    def _generate_dataset(self):
        images = []
        labels = []
        num_classes = len(self.CLASS_NAMES)
        samples_per_class = self.num_samples // num_classes

        h, w = self.image_size

        for cls_idx in range(num_classes):
            for _ in range(samples_per_class):
                panel = self._generate_panel_base()

                if cls_idx == 0:  # Healthy_Panel
                    # Uniform thermal baseline with minor grid variations
                    pass

                elif cls_idx == 1:  # Micro_Crack
                    # High-contrast fracture line across cell
                    x0, y0 = self.rng.randint(2, w - 2), self.rng.randint(2, h - 2)
                    dx = self.rng.choice([-1, 1]) * self.rng.randint(4, 10)
                    dy = self.rng.choice([-1, 1]) * self.rng.randint(4, 10)
                    x1 = np.clip(x0 + dx, 0, w - 1)
                    y1 = np.clip(y0 + dy, 0, h - 1)

                    rr, cc = np.linspace(y0, y1, 15).astype(int), np.linspace(x0, x1, 15).astype(int)
                    panel[rr, cc] = 0.05  # Dark fracture line in EL image

                elif cls_idx == 2:  # Hotspot_Fault
                    # Localized high-temperature Gaussian thermal anomaly
                    cx, cy = self.rng.randint(6, w - 6), self.rng.randint(6, h - 6)
                    sigma = self.rng.uniform(2.0, 4.5)
                    yy, xx = np.ogrid[:h, :w]
                    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
                    gaussian_spot = np.exp(-dist_sq / (2 * sigma ** 2)).astype(np.float32)
                    panel = np.clip(panel + 0.55 * gaussian_spot, 0.0, 1.0)

                elif cls_idx == 3:  # Dust_Soiling
                    # Non-uniform surface shadow mask
                    gradient = np.linspace(0.1, 0.7, w, dtype=np.float32)
                    mask = np.tile(gradient, (h, 1))
                    if self.rng.rand() > 0.5:
                        mask = mask.T
                    panel = np.clip(panel * (1.0 - 0.45 * mask), 0.0, 1.0)

                # Convert float [0, 1] to uint8 [0, 255] for PIL / transform compatibility
                img_uint8 = (panel * 255.0).astype(np.uint8)
                images.append(img_uint8)
                labels.append(cls_idx)

        # Shuffle dataset
        indices = self.rng.permutation(len(images))
        images = [images[i] for i in indices]
        labels = [labels[i] for i in indices]

        return images, np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_np = self.images[idx]
        label = self.labels[idx]
        pil_img = Image.fromarray(img_np, mode="L")

        if self.transform is not None:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).float() / 255.0

        return img_tensor, label


if __name__ == "__main__":
    ds = SolarPVFaultDataset(num_samples=100)
    print(f"[SolarPVFaultDataset] Total samples: {len(ds)}")
    img, lbl = ds[0]
    print(f"  Sample image tensor shape: {img.shape}, label: {lbl} ({SolarPVFaultDataset.CLASS_NAMES[lbl]})")
