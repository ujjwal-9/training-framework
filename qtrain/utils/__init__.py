import numpy as np
import torch
from torchvision import transforms as tfms

def fix_sitk_arr_shape(arr, req_shape=(512,512)):
    return tfms.Resize(req_shape)(torch.Tensor(arr)).numpy()

def apply_torch_transform(transform, state, input_tensor, target_tensor):
    torch.set_rng_state(state)
    input_tensor = transform(input_tensor)
    torch.set_rng_state(state)
    transform.interpolation = tfms.InterpolationMode.NEAREST
    target_tensor = transform(target_tensor)
    return input_tensor, target_tensor

def rotation_4d_metadata(dim, axis):
    axis_idx = list(range(dim))
    axis_idx.remove(axis)
    axis_idx.insert(0, axis)
    redo_axis_idx = list(range(1,dim))
    redo_axis_idx.insert(axis, 0)
    return axis_idx, redo_axis_idx

def apply_torch_transform_3d(transform, state, axis, input_tensor, target_tensor):
    axis_idx, redo_axis_idx = rotation_4d_metadata(input_tensor.shape[0]+1, axis)
    input_tensor = input_tensor.permute(axis_idx)
    for i in range(input_tensor.shape[0]):
        input_tensor[i], target_tensor_ = apply_torch_transform(transform, state, input_tensor[i], target_tensor[i].unsqueeze(0))
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
                bbox.append([0,0,0,0])
                continue
            x = ((annot[batch_][i].sum(axis=1)>0)==True).nonzero(as_tuple=True)[0]
            y = ((annot[batch_][i].sum(axis=0)>0)==True).nonzero(as_tuple=True)[0]
            try:
                bbox.append([x[0].item(), y[0].item(), x[-1].item(), y[-1].item()])
            except IndexError:
                print(x,y)
                print(annot[batch_][i].shape, x.shape, y.shape)
        bbox_batch.append(bbox)
    return torch.Tensor(bbox_batch).to(annot.device)

def crop_3d_volume(img_tensor, crop_dim, crop_size):
    assert img_tensor.ndim == 3, '3d tensor must be provided'
    (full_dim1, full_dim2, full_dim3) = img_tensor.shape
    (slices_crop, w_crop, h_crop) = crop_dim
    (dim1, dim2, dim3) = crop_size

   # check if crop size matches image dimensions
    if (full_dim2 == dim2) and (full_dim3 == dim3):
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, :, :]
    
    elif full_dim1 == dim1:
        img_tensor = img_tensor[:, w_crop:w_crop + dim2, h_crop:h_crop
                                + dim3]
    elif full_dim2 == dim2:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, :,
                                h_crop:h_crop + dim3]
    elif full_dim3 == dim3:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:
                                w_crop + dim2, :]
    else:

   # standard crop

        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:
                                w_crop + dim2, h_crop:h_crop + dim3]
    return img_tensor


def find_random_crop_dim(full_vol_dim, crop_size):
    assert full_vol_dim[0] >= crop_size[0], 'crop size is too big'
    assert full_vol_dim[1] >= crop_size[1], 'crop size is too big'
    assert full_vol_dim[2] >= crop_size[2], 'crop size is too big'

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
    crop_size= find_random_crop_dim(input_volume.shape, crop_dims)
    input_volume_ = crop_3d_volume(input_volume, crop_size, crop_dims)
    gt_volume_ = crop_3d_volume(gt_volume, crop_size, crop_dims)
    return input_volume_, gt_volume_