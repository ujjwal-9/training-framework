"""Preprocessing required for CT."""
import numpy as np
import image_transforms.transforms as tsfms
import warnings


__all__ = [
    "window_generator",
    "brain_window",
    "blood_window",
    "bone_window",
    "stroke_window",
    "soft_tissue_window",
    "WindowsAsChannelsTransform",
]


def window_generator(window_width, window_level):
    """Return CT window transform for given width and level."""
    low = window_level - window_width / 2
    high = window_level + window_width / 2

    def window_fn(img):
        img = (img - low) / (high - low)
        img = np.clip(img, 0, 1)
        return img

    return window_fn

# windows source: https://radiopaedia.org/articles/ct-head-an-approach
brain_window = window_generator(80, 40)
blood_window = window_generator(175, 50)
bone_window = window_generator(3000, 500)
stroke_window = window_generator(40, 40)
soft_tissue_window = window_generator(350, 40)


class WindowsAsChannelsTransform(tsfms.NDTransform):
    """Windows are channels."""

    def __init__(self, windows=[brain_window, blood_window, bone_window]):
        self.windows = windows

    def _transform(self, img, is_label):
        if is_label:
            return img

        if img.min() > -500:
            warnings.warn(
                f"Input dynamic range is not large. "
                f"Shape: {img.shape}, dtype: {img.dtype}, "
                f"dynamic range: {img.min()} - {img.max()}"
            )

        img = img.astype("float")
        windowed = [window(img) for window in self.windows]
        windowed = np.moveaxis(windowed, 0, -1)
        return windowed

def hu_clip(low, high):
    """Return CT window transform for given low and high hu value."""
    def clipping_fn(img):
        return np.clip(img, low, high)
    
    return clipping_fn

acute_stroke_window = hu_clip(10,25)
chronic_stroke_window = hu_clip(3,10)