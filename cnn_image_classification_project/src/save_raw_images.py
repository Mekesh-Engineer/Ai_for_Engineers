"""
save_raw_images.py
-------------------
Utility script to populate raw thermographic image files into:
data/raw/solar_pv_faults/<class_name>/

Organizes images in standard PyTorch ImageFolder layout for CNN model training:
  data/raw/solar_pv_faults/
  ├── Healthy_Panel/
  ├── Micro_Crack/
  ├── Hotspot_Fault/
  └── Dust_Soiling/
"""

import os
import sys
import numpy as np
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from eee_dataset import SolarPVFaultDataset


def save_raw_solar_pv_images(target_dir: str = "data/raw/solar_pv_faults", samples_per_class: int = 10) -> None:
    """Generate and save raw PNG thermography image files organized by class directory."""
    target_path = os.path.join(_PROJECT_ROOT, target_dir) if not os.path.isabs(target_dir) else target_dir
    os.makedirs(target_path, exist_ok=True)

    class_names = SolarPVFaultDataset.CLASS_NAMES
    print(f"[*] Exporting raw dataset images to: '{target_path}'")
    print(f"[*] Organization layout: PyTorch ImageFolder format ({len(class_names)} class subdirectories)")

    # Create subdirectories
    for c_name in class_names:
        class_dir = os.path.join(target_path, c_name)
        os.makedirs(class_dir, exist_ok=True)

    total_num = samples_per_class * len(class_names)
    ds = SolarPVFaultDataset(num_samples=total_num, seed=42)

    saved_counts = {c_name: 0 for c_name in class_names}

    for i in range(len(ds)):
        img_np = ds.images[i]
        label_idx = ds.labels[i]
        class_name = class_names[label_idx]

        count = saved_counts[class_name] + 1
        saved_counts[class_name] = count

        filename = f"{class_name.lower()}_{count:03d}.png"
        filepath = os.path.join(target_path, class_name, filename)

        # Save as grayscale PNG image
        pil_img = Image.fromarray(img_np, mode="L")
        pil_img.save(filepath)

    print("\n[+] Raw Images Successfully Created & Saved:")
    for c_name, count in saved_counts.items():
        subfolder = os.path.join(target_path, c_name)
        print(f"  - {c_name:<16} : {count} raw images saved in '{subfolder}'")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    save_raw_solar_pv_images()
