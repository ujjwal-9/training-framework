"""Preprocessing required for CT."""
import warnings
import copy
import numpy as np
import image_transforms.transforms as tsfms

from skimage.filters import gaussian


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

def hu_clip(low, high):
    """Return CT window transform for given low and high hu value."""
    def clipping_fn(img):
        return np.clip(img, low, high)
    
    return clipping_fn


def stroke_mask_generator():
    """Return CT window transform for given width and level."""

    def window_fn(img):
        acute = hu_clip(20,30)
        csf = hu_clip(0,20)
        normal = hu_clip(30,1000)

        csf_fi = csf(img)
        csf_fi[csf_fi<np.percentile(csf_fi, 82)] = 1
        csf_fi[csf_fi>=np.percentile(csf_fi, 82)] = 0

        normal_fi = normal(img)
        normal_fi[normal_fi<np.percentile(normal_fi, 95)] = 1
        normal_fi[normal_fi>np.percentile(normal_fi, 95)] = 0

        acute_fi = acute(img)
        img = csf_fi*normal_fi*img

        acute_mask_org = copy.deepcopy(img)
        img[img > np.percentile(img, 98)]=0
        img[img <= np.percentile(img, 98)]=1
        hypodense_regions = acute_mask_org*img

        hypodense_regions[(hypodense_regions > np.percentile(hypodense_regions, 97)) &\
                  (hypodense_regions < np.percentile(hypodense_regions, 98))]=0
        hypodense_regions[(hypodense_regions < np.percentile(hypodense_regions, 97)) &\
                        (hypodense_regions > np.percentile(hypodense_regions, 98))]=1

        hypodense_regions = gaussian(hypodense_regions, sigma=0.1)

        return brain_window(hypodense_regions)

    return window_fn

acute_stroke_window_extended = hu_clip(20,30)
acute_stroke_window = hu_clip(10,25)
chronic_stroke_window = hu_clip(3,10)
acute_mask_window = stroke_mask_generator()
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