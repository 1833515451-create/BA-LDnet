import albumentations as A
import cv2
import numpy as np


def build_domain_transforms(img_size):
    return A.Compose([
        A.OneOf([A.CropNonEmptyMaskIfExists(height=max(96, int(img_size * 0.75)), width=max(96, int(img_size * 0.75)), p=1.0), A.NoOp(p=1.0)], p=0.25),
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.20, rotate_limit=25, border_mode=cv2.BORDER_REFLECT_101, p=0.65),
        A.OneOf([
            A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
            A.RandomGamma(gamma_limit=(70, 140), p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=22, val_shift_limit=18, p=1.0),
            A.RGBShift(r_shift_limit=18, g_shift_limit=18, b_shift_limit=18, p=1.0),
        ], p=0.85),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussNoise(var_limit=(8.0, 40.0), p=1.0),
            A.ImageCompression(quality_lower=45, quality_upper=95, p=1.0),
        ], p=0.35),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.18),
    ])


def build_preprocess_fn(encoder_name, encoder_weights):
    import segmentation_models_pytorch as smp
    if encoder_weights and encoder_weights.lower() != "none":
        preprocess_input = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)
        return lambda image: preprocess_input(image)
    return lambda image: image.astype(np.float32) / 255.0
