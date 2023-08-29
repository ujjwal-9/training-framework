import inspect
import torch
import numpy as np
from torchvision import transforms as tfms
from collections import defaultdict


class RandomCrop:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    @staticmethod
    def get_params(tensor, output_size):
        _, _, h, w = tensor.size()
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        # Applying the same random crop to both image and mask
        i, j, h, w = self.get_params(image, self.output_size)

        image = image[:, :, i : i + h, j : j + w]
        mask = mask[:, i : i + h, j : j + w]

        return image, mask


def hu_windowing(window_configs):
    window_widths, window_levels = window_configs[:, 0], window_configs[:, 1]

    def do_windowing(scan):
        scan = scan.unsqueeze(1).expand(-1, window_configs.shape[0], -1, -1)
        # Calculate the HU limits based on the window widths and levels
        hu_mins = window_levels.unsqueeze(1) - window_widths.unsqueeze(1) / 2
        hu_maxs = window_levels.unsqueeze(1) + window_widths.unsqueeze(1) / 2

        # Clip the HU values to the specified ranges
        scan = torch.clamp(scan, hu_mins[..., None], hu_maxs[..., None])

        # Normalize the HU values to the range [0, 1]
        scan = (scan - hu_mins[..., None]) / (hu_maxs[..., None] - hu_mins[..., None])
        return scan

    return do_windowing


def default_factory(inner_default_type=list):
    return defaultdict(inner_default_type)


def get_sens_spec_youden(confusion_matrix):
    sens = (confusion_matrix["tp"] + 1e-6) / (
        confusion_matrix["tp"] + confusion_matrix["fn"] + 1e-6
    )
    spec = (confusion_matrix["tn"] + 1e-6) / (
        confusion_matrix["tn"] + confusion_matrix["fp"] + 1e-6
    )
    youden = sens + spec - 1
    return sens, spec, youden


def is_from_torchmetrics(func):
    if not callable(func):
        return False
    module_name = inspect.getmodule(func).__name__
    return module_name.startswith("torchmetrics")


def put_torchmetric_to_device(func, device):
    if is_from_torchmetrics(func):
        func.to(device)
    return func


def fix_sitk_arr_shape(arr, req_shape=(512, 512)):
    return tfms.Resize(req_shape)(torch.Tensor(arr)).numpy()


def apply_torch_transform(transform, state, input_tensor, target_tensor=None):
    torch.set_rng_state(state)
    input_tensor = transform(input_tensor)
    if target_tensor is None:
        return input_tensor, None
    torch.set_rng_state(state)
    transform.interpolation = tfms.InterpolationMode.NEAREST
    target_tensor = transform(target_tensor)
    return input_tensor, target_tensor


def rotation_4d_metadata(dim, axis):
    axis_idx = list(range(dim))
    axis_idx.remove(axis)
    axis_idx.insert(0, axis)
    redo_axis_idx = list(range(1, dim))
    redo_axis_idx.insert(axis, 0)
    return axis_idx, redo_axis_idx


def apply_torch_transform_3d(transform, state, axis, input_tensor, target_tensor=None):
    axis_idx, redo_axis_idx = rotation_4d_metadata(input_tensor.shape[0] + 1, axis)
    input_tensor = input_tensor.permute(axis_idx)
    for i in range(input_tensor.shape[0]):
        target_tensor_ = (
            target_tensor[i].unsqueeze(0) if target_tensor is not None else None
        )
        input_tensor[i], target_tensor_ = apply_torch_transform(
            transform, state, input_tensor[i], target_tensor_
        )
        if target_tensor is not None:
            target_tensor[i] = target_tensor_[0]
    input_tensor = input_tensor.permute(redo_axis_idx)
    return input_tensor, target_tensor


def get_bounding_box(annot=None):
    if annot is None:
        raise ValueError("Annotation are None, check code ;), or your groundtruths")
    bbox_batch = []
    for batch_ in range(annot.shape[0]):
        bbox = []
        for i in range(annot.shape[1]):
            if annot[batch_][i].sum() == 0:
                bbox.append([0, 0, 0, 0])
                continue
            x = ((annot[batch_][i].sum(axis=1) > 0) == True).nonzero(as_tuple=True)[0]
            y = ((annot[batch_][i].sum(axis=0) > 0) == True).nonzero(as_tuple=True)[0]
            try:
                bbox.append([x[0].item(), y[0].item(), x[-1].item(), y[-1].item()])
            except IndexError:
                print(x, y)
                print(annot[batch_][i].shape, x.shape, y.shape)
        bbox_batch.append(bbox)
    return torch.Tensor(bbox_batch).to(annot.device)


def crop_3d_volume(img_tensor, crop_dim, crop_size):
    assert img_tensor.ndim == 3, "3d tensor must be provided"
    (full_dim1, full_dim2, full_dim3) = img_tensor.shape
    (slices_crop, w_crop, h_crop) = crop_dim
    (dim1, dim2, dim3) = crop_size

    # check if crop size matches image dimensions
    if (full_dim2 == dim2) and (full_dim3 == dim3):
        img_tensor = img_tensor[slices_crop : slices_crop + dim1, :, :]

    elif full_dim1 == dim1:
        img_tensor = img_tensor[:, w_crop : w_crop + dim2, h_crop : h_crop + dim3]
    elif full_dim2 == dim2:
        img_tensor = img_tensor[
            slices_crop : slices_crop + dim1, :, h_crop : h_crop + dim3
        ]
    elif full_dim3 == dim3:
        img_tensor = img_tensor[
            slices_crop : slices_crop + dim1, w_crop : w_crop + dim2, :
        ]
    else:
        # standard crop

        img_tensor = img_tensor[
            slices_crop : slices_crop + dim1,
            w_crop : w_crop + dim2,
            h_crop : h_crop + dim3,
        ]
    return img_tensor


def find_random_crop_dim(full_vol_dim, crop_size):
    assert full_vol_dim[0] >= crop_size[0], "crop size is too big"
    assert full_vol_dim[1] >= crop_size[1], "crop size is too big"
    assert full_vol_dim[2] >= crop_size[2], "crop size is too big"

    if full_vol_dim[0] == crop_size[0]:
        slices = crop_size[0]
    else:
        slices = np.random.randint(full_vol_dim[0] - crop_size[0])

    if full_vol_dim[1] == crop_size[1]:
        w_crop = crop_size[1]
    else:
        w_crop = np.random.randint(full_vol_dim[1] - crop_size[1])

    if full_vol_dim[2] == crop_size[2]:
        h_crop = crop_size[2]
    else:
        h_crop = np.random.randint(full_vol_dim[2] - crop_size[2])

    return (slices, w_crop, h_crop)


def get_croped_3d_volume(input_volume, gt_volume, crop_dims):
    crop_size = find_random_crop_dim(input_volume.shape, crop_dims)
    input_volume_ = crop_3d_volume(input_volume, crop_size, crop_dims)
    if gt_volume is None:
        return input_volume_, None
    gt_volume_ = crop_3d_volume(gt_volume, crop_size, crop_dims)
    return input_volume_, gt_volume_
