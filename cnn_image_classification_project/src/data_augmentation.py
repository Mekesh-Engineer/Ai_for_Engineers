"""
data_augmentation.py
--------------------
DataAugmentationModule: Defines data augmentation pipelines for solar thermography
training set generalization and un-augmented transforms for validation/testing.
"""

def get_transforms(config: dict):
    """
    Construct torchvision dataset transforms for training and validation/testing.

    Parameters
    ----------
    config : dict
        Hyperparameters configuration dictionary.

    Returns
    -------
    tuple (train_transform, eval_transform)
        PyTorch torchvision transform pipelines.
    """
    import torchvision.transforms as T

    mean = config["preprocessing"]["normalize_mean"]
    std  = config["preprocessing"]["normalize_std"]
    aug_cfg = config.get("augmentation", {})

    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((mean,), (std,)),
    ])

    if aug_cfg.get("enable", True):
        rot_deg   = aug_cfg.get("rotation_degrees", 15)
        translate = tuple(aug_cfg.get("affine_translate", [0.05, 0.05]))
        pad       = aug_cfg.get("random_crop_padding", 2)

        transform_list = [
            T.RandomCrop(28, padding=pad, padding_mode="edge"),
            T.RandomAffine(degrees=rot_deg, translate=translate),
        ]
        if aug_cfg.get("horizontal_flip", True):
            transform_list.append(T.RandomHorizontalFlip(p=0.5))
        if aug_cfg.get("vertical_flip", True):
            transform_list.append(T.RandomVerticalFlip(p=0.5))

        transform_list.extend([
            T.ToTensor(),
            T.Normalize((mean,), (std,)),
        ])

        train_transform = T.Compose(transform_list)
    else:
        train_transform = eval_transform

    return train_transform, eval_transform


if __name__ == "__main__":
    dummy_cfg = {
        "preprocessing": {"normalize_mean": 0.35, "normalize_std": 0.15},
        "augmentation": {"enable": True, "rotation_degrees": 15, "affine_translate": [0.05, 0.05], "random_crop_padding": 2}
    }
    try:
        t_train, t_eval = get_transforms(dummy_cfg)
        print("[DataAugmentation] Transforms successfully created:")
        print("  Train transform:", t_train)
        print("  Eval transform :", t_eval)
    except ModuleNotFoundError:
        print("[DataAugmentation] torchvision not yet available.")
