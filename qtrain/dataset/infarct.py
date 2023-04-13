import os
import json
import PIL
import glob
import pydicom
import random
import platform
import numpy as np
import pandas as pd
import SimpleITK as sitk


import torch
import pytorch_lightning as pl


from qtrain.utils import windowing
from torchvision import transforms
from monai.transforms import CropForeground
from torch.utils.data import Dataset, DataLoader


from tqdm import tqdm
from warnings import warn
from qtrain.utils import apply_torch_transform_3d, get_croped_3d_volume


class InfarctDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.num_workers and platform.system() == "Windows":
            # see: https://stackoverflow.com/a/59680818
            warn(
                f"You have requested num_workers={self.args.num_workers} on Windows,"
                " but currently recommended is 0, so we set it for you"
            )
            self.args.num_workers = 0

        self.num_workers = self.args.num_workers
        self.seed = self.args.seed
        self.batch_size = self.args.batch_size
        self.val_batch_size = self.args.val_batch_size

    def num_classes(self):
        return self.args.num_classes

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """Split the train, valid and test dataset"""
        pass
            
    def setup_dataset(self, status):
        dataset = self.args.dataset_class(self.args, status)
        return dataset

    def setup_dataloader(self, dataset, batch_size):
        sampler = None
        shuffle = True
        if self.args.sampler is not None:
            shuffle = None
            sampler = self.args.sampler(dataset)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle= shuffle,
            sampler=sampler
        )
        return loader
    
    def train_dataloader(self):
        train_dataset = self.setup_dataset("train")
        self.num_classes = train_dataset.max_classes
        loader = self.setup_dataloader(train_dataset, self.batch_size)
        return loader

    def val_dataloader(self):
        loader = self.setup_dataloader(self.setup_dataset("valid"), self.val_batch_size)
        return loader

    def test_dataloader(self):
        pass

class InfarctDataset3D(Dataset):
    def __init__(self, args, run_type="train", init_dataset=True):
        self.args = args   
        self.run_type = run_type
        self.input_shape = [self.args.img_size, self.args.img_size]
        self.window_as_channel = self.window_channels()
        if init_dataset:
            self.initialize_dataset(self.args.datapath)

    def get_studyuid(self, index):
        return self.dataset.loc[index,"series"].values[0]
    
    def append_dataset(self, scan, annotation, label):
        self.datapath.append(scan)
        self.targetpath.append(annotation)
        self.labels.append(label)
    
    def initialize_dataset(self, datapath):
        with open(datapath, "r") as f:
            self.dataset = pd.DataFrame(json.load(f))
        
        self.dataset = self.dataset[self.dataset['status'] == self.run_type].reset_index(drop=True)

        self.datapath = []
        self.targetpath = []
        self.labels = []
        
        if self.run_type in ['train', 'test', 'valid']:
            if self.args.fast_dev_run == True:
                counter = self.args.fast_batch_size
            else:
                counter = len(self.dataset)
            
            self.max_classes = 0
            for index in tqdm(range(counter), desc=f"{self.run_type} data"):
                filepath, annotpath, labels = self.dataset.loc[index,"filepath"], self.dataset.loc[index,"annotpath"], self.dataset.loc[index,"labels"]
                n_classes = len(labels)
                if n_classes > self.max_classes:
                    self.max_classes = n_classes
                if self.args.only_annotated:
                    if_annotated = False
                    for key in labels:
                        if n_classes > 0:
                            if_annotated = True
                    if if_annotated:
                        self.append_dataset(filepath, annotpath, labels)
                elif self.args.only_annotated == False:
                    self.append_dataset(filepath, annotpath, labels)
        if self.args.mode == "multiclass":
            self.max_classes += 1
        print(f"{len(self.datapath)} scans present in {self.run_type} data belonging to {self.max_classes} classes\n")


    def get_crop(self, scan, annot):
        return get_croped_3d_volume(scan, annot, [self.args.n_slices]+self.input_shape)
    
    def initialize_transform(self, annot=False):
        if annot:
            interpolation = transforms.InterpolationMode.NEAREST
        else:
            interpolation = transforms.InterpolationMode.BILINEAR
            
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize(self.input_shape, interpolation)
                                  ])
    
    
    def window_channels(self):
        # Take stroke window instead of bone window(add as an arguments)
        # self.window_as_channel = windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window, windowing.blood_window, windowing.stroke_window])
        if self.args.windowing == "old":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window, 
                                                             windowing.blood_window, 
                                                             windowing.stroke_window])
        elif self.args.windowing == "brain":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window])

        elif self.args.windowing == "brain-infarcts":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window,
                                                                windowing.acute_stroke_window,
                                                                windowing.chronic_stroke_window])
        elif self.args.windowing == "gray-white-matter":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window,
                                                                windowing.acute_stroke_window,
                                                                windowing.chronic_stroke_window])

        elif self.args.windowing == "stroke-infarcts":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.stroke_window,
                                                                windowing.acute_stroke_window,
                                                                windowing.chronic_stroke_window])
        else:
            raise ValueError(f"Unknown windowing type: {self.args.windowing}")
            exit(0)
    
    def get_window_channels(self, scan):
        scan_windowed = [torch.tensor(self.window_as_channel(scan[i].numpy())).unsqueeze(0) for i in range(scan.shape[0])]
        scan_windowed = torch.cat(scan_windowed, dim=0) 
        if scan_windowed.shape[3] == 1:
            scan_windowed = scan_windowed.permute(3,0,1,2)
            self.spatial_dim_idx = 1
        else:
            scan_windowed = scan_windowed.permute(3,1,2,0)
            self.spatial_dim_idx = 3
        return scan_windowed
    
    def get_foreground_crop(self, sitk_arr, annot_arr):
        cropped_sitk_arr, coord_top, coord_bottom = CropForeground(return_coords=True)(sitk_arr)
        cropped_annot_arr = annot_arr[:, coord_top[0]:coord_bottom[0], coord_top[1]:coord_bottom[1]]
        cropped_annot_arr[cropped_annot_arr>0] = 1
        return np.moveaxis(cropped_sitk_arr.numpy(),0,-1), np.moveaxis(cropped_annot_arr,0,-1)
    
    def process(self, index):
        annotation = self.process_annotation_label(index)
        ct_scan = sitk.GetArrayFromImage(sitk.ReadImage(self.datapath[index]))
        ct_scan, annotation = self.get_foreground_crop(ct_scan, annotation)
        ct_scan, annotation = self.initialize_transform()(ct_scan), self.initialize_transform(True)(annotation)
        annotation[annotation>0] = 1
        annotation = annotation.to(torch.int)
        ct_scan, annotation = self.get_crop(ct_scan, annotation)
        ct_scan = self.get_window_channels(ct_scan)
        return ct_scan, annotation
    
    def process_annotation_label(self, index):
        label_keys = list(self.labels[index].keys())
        annotation = sitk.GetArrayFromImage(sitk.ReadImage(self.targetpath[index]))

        if self.args.mode == "binary":
            annotation = np.where(np.isin(annotation, self.labels[index][label_keys[0]]), 1, 0)
        
        elif self.args.mode in [ "multilabel", "multiclass"]:
            if len(label_keys) < 2:
                raise ValueError(f"Use binary mode for training")
            
            annotations = []
            for i, key in enumerate(label_keys):
                if self.args.mode == "multilabel":
                    annotations.append(np.where(np.isin(annotation, self.labels[index][key]), 1, 0))
                elif self.args.mode == "multiclass":
                    annotations.append(np.where(np.isin(annotation, self.labels[index][key]), i+1, 0))

            if self.args.mode == "multilabel":
                annotation = np.stack(annotations)
            elif self.args.mode == "multiclass":
                annotation = np.maximum.reduce(annotations)
            
        return annotation
        
    def excute_augmentations(self, ct_scan, annotation):
        if self.run_type == 'train' and self.args.augmentation:
            if random.random() < 0.35:
                ct_scan, annotation = self.train_transforms(ct_scan, annotation) 
        return ct_scan, annotation

    def __getitem__(self, index):
        ct_scan, annotation = self.process(index)
        return self.excute_augmentations(ct_scan, annotation)

    def __len__(self):
        return len(self.datapath)
    
    def train_transforms(self, input_scan, target):
        seed_ = random.random()
        state = torch.get_rng_state()
        if not self.args.extra_augs:
            if seed_ < 0.6:
                random_rotate = transforms.RandomAffine(scale=(0.9,1.3), degrees=(-45,45), interpolation=transforms.InterpolationMode.BILINEAR)
                if self.args.dataset_type == "3D":
                    input_scan, target = apply_torch_transform_3d(random_rotate, state, self.spatial_dim_idx, input_scan, target)
            elif seed_ > 0.5:
                random_flip = transforms.RandomHorizontalFlip(p=1)
                if self.args.dataset_type == "3D":
                    input_scan, target = apply_torch_transform_3d(random_flip, state, self.spatial_dim_idx, input_scan, target)
        
        elif self.args.extra_augs:
            # Random transforms
            if seed_ < 0.6:
                input_scan = transforms.GaussianBlur(5, sigma=(0.1, 2.0))(input_scan)
            
        return input_scan, target


class InfarctDataset3D_only_cls(Dataset):
    def __init__(self, args, run_type="train", init_dataset=True, crop=True):
        self.args = args   
        self.run_type = run_type
        self.crop = crop
        self.input_shape = [self.args.img_size, self.args.img_size]
        self.window_as_channel = self.window_channels()
        if init_dataset:
            self.initialize_dataset(self.args.datapath)

    def get_studyuid(self, index):
        return self.dataset.loc[index,"series"]
    
    def append_dataset(self, scan, annotation, label):
        self.datapath.append(scan)
        self.targetpath.append(annotation)
        self.labels.append(label)
    
    def initialize_dataset(self, datapath):
        with open(datapath, "r") as f:
            self.dataset = pd.DataFrame(json.load(f))
        
        self.dataset = self.dataset[self.dataset['status'] == self.run_type].reset_index(drop=True)

        self.datapath = []
        self.targetpath = []
        self.labels = []
        
        if self.run_type in ['train', 'test', 'valid']:
            if self.args.fast_dev_run == True:
                counter = self.args.fast_batch_size
            else:
                counter = len(self.dataset)
            
            self.max_classes = 0
            for index in tqdm(range(counter), desc=f"{self.run_type} data"):
                filepath, annotpath, labels = self.dataset.loc[index,"filepath"], self.dataset.loc[index,"annotpath"], self.dataset.loc[index,"labels"]
                n_classes = len(labels)
                if n_classes > self.max_classes:
                    self.max_classes = n_classes
                self.append_dataset(filepath, annotpath, labels)
        if self.args.mode == "multiclass":
            self.max_classes += 1
        print(f"{len(self.datapath)} scans present in {self.run_type} data belonging to {self.max_classes} classes\n")


    def get_crop(self, scan, annot):
        return get_croped_3d_volume(scan, annot, [self.args.n_slices]+self.input_shape)
    
    def initialize_transform(self, annot=False):
        if annot:
            interpolation = transforms.InterpolationMode.NEAREST
        else:
            interpolation = transforms.InterpolationMode.BILINEAR
            
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize(self.input_shape, interpolation)
                                  ])
    
    
    def window_channels(self):
        # Take stroke window instead of bone window(add as an arguments)
        # self.window_as_channel = windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window, windowing.blood_window, windowing.stroke_window])
        if self.args.windowing == "old":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window, 
                                                             windowing.blood_window, 
                                                             windowing.bone_window])
        elif self.args.windowing == "brain":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window])

        elif self.args.windowing == "brain-infarcts":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window,
                                                                windowing.acute_stroke_window,
                                                                windowing.chronic_stroke_window])
        elif self.args.windowing == "gray-white-matter":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window,
                                                                windowing.acute_stroke_window,
                                                                windowing.chronic_stroke_window])

        elif self.args.windowing == "stroke-infarcts":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.stroke_window,
                                                                windowing.acute_stroke_window,
                                                                windowing.chronic_stroke_window])

        elif self.args.windowing == "stroke-masks":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window,
                                                                windowing.acute_mask_window,
                                                                windowing.acute_stroke_window_extended])
        else:
            raise ValueError(f"Unknown windowing type: {self.args.windowing}")
            exit(0)
    
    def get_window_channels(self, scan):
        scan_windowed = [torch.tensor(self.window_as_channel(scan[i].numpy())).unsqueeze(0) for i in range(scan.shape[0])]
        scan_windowed = torch.cat(scan_windowed, dim=0) 
        if scan_windowed.shape[3] == 1:
            scan_windowed = scan_windowed.permute(3,0,1,2)
            self.spatial_dim_idx = 1
        else:
            scan_windowed = scan_windowed.permute(3,1,2,0)
            self.spatial_dim_idx = 3
        return scan_windowed
    
    def get_foreground_crop(self, sitk_arr):
        cropped_sitk_arr = CropForeground()(sitk_arr)
        return np.moveaxis(cropped_sitk_arr.numpy(),0,-1)
    
    def process(self, index):
        annotation = self.process_annotation_label(index)
        ct_scan = sitk.GetArrayFromImage(sitk.ReadImage(self.datapath[index]))
        ct_scan = self.get_foreground_crop(ct_scan)
        ct_scan = self.initialize_transform()(ct_scan)
        if ct_scan.shape[0] <= self.args.n_slices:
            ct_scan = torch.cat([ct_scan, torch.zeros(self.args.n_slices - ct_scan.shape[0] + 1, *ct_scan.shape[1:])])
        if self.crop:
            ct_scan, _ = self.get_crop(ct_scan, None)
        ct_scan = self.get_window_channels(ct_scan)
        return ct_scan, annotation
    
    def process_annotation_label(self, index):
        labels = list(self.labels[index].values())[0]
        annotation = 0
        if len(labels) > 0:
            annotation = 1
        return annotation
        
    def excute_augmentations(self, ct_scan):
        if self.run_type == 'train' and self.args.augmentation:
            if random.random() < 0.35:
                ct_scan, annotation = self.train_transforms(ct_scan) 
        ct_scan = ct_scan.permute(3,0,1,2)
        return ct_scan

    def __getitem__(self, index):
        ct_scan, annotation = self.process(index)
        return self.excute_augmentations(ct_scan), annotation

    def __len__(self):
        return len(self.datapath)
    
    def train_transforms(self, input_scan, target):
        seed_ = random.random()
        state = torch.get_rng_state()
        if not self.args.extra_augs:
            if seed_ < 0.6:
                random_rotate = transforms.RandomAffine(scale=(0.9,1.3), degrees=(-45,45), interpolation=transforms.InterpolationMode.BILINEAR)
                if self.args.dataset_type == "3D":
                    input_scan, target = apply_torch_transform_3d(random_rotate, state, self.spatial_dim_idx, input_scan, target=None)
            elif seed_ > 0.5:
                random_flip = transforms.RandomHorizontalFlip(p=1)
                if self.args.dataset_type == "3D":
                    input_scan, target = apply_torch_transform_3d(random_flip, state, self.spatial_dim_idx, input_scan, target=None)
        
        elif self.args.extra_augs:
            # Random transforms
            if seed_ < 0.6:
                input_scan = transforms.GaussianBlur(5, sigma=(0.1, 2.0))(input_scan)
            
        return input_scan



class InfarctDataset3D_cls(Dataset):
    def __init__(self, args, run_type="train", init_dataset=True, crop=True):
        self.args = args   
        self.run_type = run_type
        self.crop = crop
        self.input_shape = [self.args.img_size, self.args.img_size]
        self.window_as_channel = self.window_channels()
        if init_dataset:
            self.initialize_dataset(self.args.datapath)

    def get_studyuid(self, index):
        return self.dataset.loc[index,"series"]
    
    def append_dataset(self, scan, annotation, label):
        self.datapath.append(scan)
        self.targetpath.append(annotation)
        self.labels.append(label)
    
    def initialize_dataset(self, datapath):
        with open(datapath, "r") as f:
            self.dataset = pd.DataFrame(json.load(f))
        
        self.dataset = self.dataset[self.dataset['status'] == self.run_type].reset_index(drop=True)

        self.datapath = []
        self.targetpath = []
        self.labels = []
        
        if self.run_type in ['train', 'test', 'valid']:
            if self.args.fast_dev_run == True:
                counter = self.args.fast_batch_size
            else:
                counter = len(self.dataset)
            
            self.max_classes = 0
            for index in tqdm(range(counter), desc=f"{self.run_type} data"):
                filepath, annotpath, labels = self.dataset.loc[index,"filepath"], self.dataset.loc[index,"annotpath"], self.dataset.loc[index,"labels"]
                n_classes = len(labels)
                if n_classes > self.max_classes:
                    self.max_classes = n_classes
                self.append_dataset(filepath, annotpath, labels)
        if self.args.mode == "multiclass":
            self.max_classes += 1
        print(f"{len(self.datapath)} scans present in {self.run_type} data belonging to {self.max_classes} classes\n")


    def get_crop(self, scan, annot):
        return get_croped_3d_volume(scan, annot, [self.args.n_slices]+self.input_shape)
    
    def initialize_transform(self, annot=False):
        if annot:
            interpolation = transforms.InterpolationMode.NEAREST
        else:
            interpolation = transforms.InterpolationMode.BILINEAR
            
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize(self.input_shape, interpolation)
                                  ])
    
    
    def window_channels(self):
        # Take stroke window instead of bone window(add as an arguments)
        # self.window_as_channel = windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window, windowing.blood_window, windowing.stroke_window])
        if self.args.windowing == "old":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window, 
                                                             windowing.blood_window, 
                                                             windowing.bone_window])
        elif self.args.windowing == "brain":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window])

        elif self.args.windowing == "brain-infarcts":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window,
                                                                windowing.acute_stroke_window,
                                                                windowing.chronic_stroke_window])
        elif self.args.windowing == "gray-white-matter":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window,
                                                                windowing.acute_stroke_window,
                                                                windowing.chronic_stroke_window])

        elif self.args.windowing == "stroke-infarcts":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.stroke_window,
                                                                windowing.acute_stroke_window,
                                                                windowing.chronic_stroke_window])

        elif self.args.windowing == "stroke-masks":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window,
                                                                windowing.acute_mask_window,
                                                                windowing.acute_stroke_window_extended])
        else:
            raise ValueError(f"Unknown windowing type: {self.args.windowing}")
            exit(0)
    
    def get_window_channels(self, scan):
        scan_windowed = [torch.tensor(self.window_as_channel(scan[i].numpy())).unsqueeze(0) for i in range(scan.shape[0])]
        scan_windowed = torch.cat(scan_windowed, dim=0) 
        if scan_windowed.shape[3] == 1:
            scan_windowed = scan_windowed.permute(3,0,1,2)
            self.spatial_dim_idx = 1
        else:
            scan_windowed = scan_windowed.permute(3,1,2,0)
            self.spatial_dim_idx = 3
        return scan_windowed
    
    def get_foreground_crop(self, sitk_arr, annot_arr):
        cropped_sitk_arr, coord_top, coord_bottom = CropForeground(return_coords=True)(sitk_arr)
        cropped_annot_arr = annot_arr[:, coord_top[0]:coord_bottom[0], coord_top[1]:coord_bottom[1]]
        cropped_annot_arr[cropped_annot_arr>0] = 1
        return np.moveaxis(cropped_sitk_arr.numpy(),0,-1), np.moveaxis(cropped_annot_arr,0,-1)
    
    def process(self, index):
        annotation = self.process_annotation_label(index)
        ct_scan = sitk.GetArrayFromImage(sitk.ReadImage(self.datapath[index]))
        ct_scan, annotation = self.get_foreground_crop(ct_scan, annotation)
        ct_scan, annotation = self.initialize_transform()(ct_scan), self.initialize_transform(True)(annotation)
        annotation[annotation>0] = 1
        annotation = annotation.to(torch.int)
        if ct_scan.shape[0] <= self.args.n_slices:
            ct_scan = torch.cat([ct_scan, torch.zeros(self.args.n_slices - ct_scan.shape[0] + 1, *ct_scan.shape[1:])])
            annotation = torch.cat([annotation, torch.zeros(self.args.n_slices - annotation.shape[0] + 1, *annotation.shape[1:])])
        if self.crop:
            ct_scan, annotation = self.get_crop(ct_scan, annotation)
        ct_scan = self.get_window_channels(ct_scan)
        return ct_scan, annotation
    
    def process_annotation_label(self, index):
        label_keys = list(self.labels[index].keys())
        annotation = sitk.GetArrayFromImage(sitk.ReadImage(self.targetpath[index]))

        if self.args.mode == "binary":
            annotation = np.where(np.isin(annotation, self.labels[index][label_keys[0]]), 1, 0)
        
        elif self.args.mode in [ "multilabel", "multiclass"]:
            if len(label_keys) < 2:
                raise ValueError(f"Use binary mode for training")
            
            annotations = []
            for i, key in enumerate(label_keys):
                if self.args.mode == "multilabel":
                    annotations.append(np.where(np.isin(annotation, self.labels[index][key]), 1, 0))
                elif self.args.mode == "multiclass":
                    annotations.append(np.where(np.isin(annotation, self.labels[index][key]), i+1, 0))

            if self.args.mode == "multilabel":
                annotation = np.stack(annotations)
            elif self.args.mode == "multiclass":
                annotation = np.maximum.reduce(annotations)
            
        return annotation
        
    def excute_augmentations(self, ct_scan):
        if self.run_type == 'train' and self.args.augmentation:
            if random.random() < 0.35:
                ct_scan, annotation = self.train_transforms(ct_scan) 
        ct_scan = ct_scan.permute(3,0,1,2)
        return ct_scan

    def __getitem__(self, index):
        ct_scan, annotation = self.process(index)
        if bool((annotation.sum() > 0).item()):
            annotation = 1
        else:
            annotation = 0
        return self.excute_augmentations(ct_scan), annotation

    def __len__(self):
        return len(self.datapath)
    
    def train_transforms(self, input_scan, target):
        seed_ = random.random()
        state = torch.get_rng_state()
        if not self.args.extra_augs:
            if seed_ < 0.6:
                random_rotate = transforms.RandomAffine(scale=(0.9,1.3), degrees=(-45,45), interpolation=transforms.InterpolationMode.BILINEAR)
                if self.args.dataset_type == "3D":
                    input_scan, target = apply_torch_transform_3d(random_rotate, state, self.spatial_dim_idx, input_scan, target=None)
            elif seed_ > 0.5:
                random_flip = transforms.RandomHorizontalFlip(p=1)
                if self.args.dataset_type == "3D":
                    input_scan, target = apply_torch_transform_3d(random_flip, state, self.spatial_dim_idx, input_scan, target=None)
        
        elif self.args.extra_augs:
            # Random transforms
            if seed_ < 0.6:
                input_scan = transforms.GaussianBlur(5, sigma=(0.1, 2.0))(input_scan)
            
        return input_scan

class InfarctDataset2D_Contrast(Dataset):
    def __init__(self, args, run_type="train", init_dataset=True):
        self.args = args
        self.run_type = run_type
        self.input_shape = [self.args.img_size, self.args.img_size]
        self.window_as_channel = self.window_channels()
        if init_dataset:
            self.initialize_dataset(self.args.datapath)

    def get_studyuid(self, index):
        return self.dataset.loc[index,"series"].values[0]
    
    def get_series_path(self, series_path):
        return "/".join(series_path.split("/")[:-1])
    
    def get_series_from_path(self, series_path):
        return series_path.split("/")[-2]
    
    def get_slice_index(self, series_path):
        return int(series_path.split("/")[-1].split("_")[1].split(".")[0])
        
    def get_max_slices(self, series):
        return self.dataset.loc[self.dataset.series==series, "max_slice"].values[0]
    
    def initialize_dataset(self, datapath):
        with open(datapath, "r") as f:
            self.dataset = pd.DataFrame(json.load(f))
        
        self.dataset = self.dataset[self.dataset['status'] == self.run_type].reset_index(drop=True)
        self.positive_scans = []
        self.negative_scans = []
        self.max_slices = []
        
        if self.run_type in ['train', 'test', 'valid']:
            if self.args.fast_dev_run == True:
                counter = self.args.fast_batch_size
            else:
                counter = len(self.dataset)
            
            self.max_classes = 2
            for index in tqdm(range(counter), desc=f"{self.run_type} data"):
                series, series_path, pos_index, neg_index, max_slice = (self.dataset.loc[index,"series"], 
                                                                        self.dataset.loc[index,"series_path"], 
                                                                        self.dataset.loc[index,"pos_index"], 
                                                                        self.dataset.loc[index,"neg_index"],
                                                                        self.dataset.loc[index,"max_slice"])
                
                for i in pos_index:
                    i_name = '0'*(3-len(str(i)))+str(i)
                    self.positive_scans.append(os.path.join(series_path, f"slice_{i_name}.dcm"))
                for i in neg_index:
                    i_name = '0'*(3-len(str(i)))+str(i)
                    self.negative_scans.append(os.path.join(series_path, f"slice_{i_name}.dcm"))  
                
        if self.args.mode == "multiclass":
            self.max_classes += 1
        print(f"{len(self.positive_scans)} positive slices and {len(self.negative_scans)} negative slices, present in {self.run_type} data.\n")
    
    def initialize_transform(self, annot=False):
        if annot:
            interpolation = transforms.InterpolationMode.NEAREST
        else:
            interpolation = transforms.InterpolationMode.BILINEAR
            
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize(self.input_shape, interpolation)
                                  ])
    
    def window_channels(self):
        # Take stroke window instead of bone window(add as an arguments)
        # self.window_as_channel = windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window, windowing.blood_window, windowing.stroke_window])
        if self.args.windowing == "old":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window, 
                                                             windowing.blood_window, 
                                                             windowing.stroke_window])
        elif self.args.windowing == "brain":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window])

        elif self.args.windowing == "brain-infarcts":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window,
                                                                windowing.acute_stroke_window,
                                                                windowing.chronic_stroke_window])
        elif self.args.windowing == "gray-white-matter":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window,
                                                                windowing.acute_stroke_window,
                                                                windowing.chronic_stroke_window])

        elif self.args.windowing == "stroke-infarcts":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.stroke_window,
                                                                windowing.acute_stroke_window,
                                                                windowing.chronic_stroke_window])
        else:
            raise ValueError(f"Unknown windowing type: {self.args.windowing}")
            exit(0)
    
    def get_window_channels(self, scan):
        scan_windowed = [torch.tensor(self.window_as_channel(scan[i].numpy())).unsqueeze(0) for i in range(scan.shape[0])]
        scan_windowed = torch.cat(scan_windowed, dim=0) 
        if scan_windowed.shape[3] == 1:
            scan_windowed = scan_windowed.permute(3,0,1,2)
            self.spatial_dim_idx = 1
        else:
            scan_windowed = scan_windowed.permute(3,1,2,0)
            self.spatial_dim_idx = 3
        return scan_windowed
    
    def get_scan_channels(self, scan):
        index = self.get_slice_index(scan)
        index_1 = str(index-1)
        index_2 = str(index+1)
        index_1 = "0"*(3-len(index_1))+index_1
        index_2 = "0"*(3-len(index_2))+index_2
        series_path = self.get_series_path(scan)
        scan_b = pydicom.read_file(scan).pixel_array
        scan_a = pydicom.read_file(os.path.join(series_path, f"slice_{index_1}.dcm")).pixel_array
        scan_c = pydicom.read_file(os.path.join(series_path, f"slice_{index_2}.dcm")).pixel_array
        scan = np.stack([scan_a, scan_b, scan_c])
        return scan
            
        
    def get_foreground_crop(self, sitk_arr):
        cropped_sitk_arr = CropForeground(return_coords=False)(sitk_arr)
        return np.moveaxis(cropped_sitk_arr.numpy(),0,-1)
    
    def preprocess(self, scan):
        scan = self.get_scan_channels(scan)
        scan = self.get_foreground_crop(scan)
        scan = self.initialize_transform()(scan)
        return scan
    
    def process(self, scana, scanb):
        scana = self.preprocess(scana)
        scanb = self.preprocess(scanb)
        return scana, scanb
        
    def excute_augmentations(self, scana, scanb):
        if self.run_type == 'train' and self.args.augmentation:
            if random.random() < 0.35:
                scana = self.train_transforms(scana)
            if random.random() < 0.35:
                scanb = self.train_transforms(scanb)
        return scana, scanb

    def __getitem__(self, index):
        tossa = random.random() #selecting if pool is same
        tossb = random.random() #selecting when pool is same if it should be positive or negative
        if tossa <= 0.5:
            if tossb <= 0.5:
                scana, scanb = random.choice(self.positive_scans), random.choice(self.positive_scans)
                flag = 1
            else:
                scana, scanb = random.choice(self.negative_scans), random.choice(self.negative_scans)
                flag = 1
        else:
            scana = random.choice(self.positive_scans)
            scanb = random.choice(self.negative_scans)
            flag = 0
        scana, scanb = self.process(scana, scanb)
        return self.excute_augmentations(scana, scanb), flag

    def __len__(self):
        return int(1e7)
    
    def train_transforms(self, scana):
        seed_ = random.random()
        if not self.args.extra_augs:
            if seed_ < 0.6:
                random_rotate = transforms.RandomAffine(scale=(0.9,1.3), degrees=(-45,45), interpolation=transforms.InterpolationMode.BILINEAR)
                scana = random_rotate(scana)
            elif seed_ > 0.5:
                random_flip = transforms.RandomHorizontalFlip(p=1)
                scana = random_flip(scana)
        
        elif self.args.extra_augs:
            # Random transforms
            if seed_ < 0.6:
                gaussian_blurr = transforms.GaussianBlur(5, sigma=(0.1, 2.0))
                scana = gaussian_blurr(scana)
        return scana

class InfarctDataset2D(Dataset):
    def __init__(self, args, run_type="train", init_dataset=True):
        self.args = args   
        self.run_type = run_type
        self.input_shape = [self.args.img_size, self.args.img_size]
        if init_dataset:
            self.initialize_dataset(self.args.datapath)

    def get_studyuid(self, index):
        return self.dataset.loc[index,"series"].values[0]
    
    def get_series_path(self, series_path):

        return "/".join(series_path.split("/")[:-1])
    
    def get_series_from_path(self, series_path):
        return series_path.split("/")[-2]
    
    def get_slice_index(self, series_path):
        return int(series_path.split("/")[-1].split("_")[1].split(".")[0])
        
    def get_max_slices(self, series):
        return self.dataset.loc[self.dataset.series==series, "max_slice"].values[0]
    
    def initialize_dataset(self, datapath):
        with open(datapath, "r") as f:
            self.dataset = pd.DataFrame(json.load(f))
        
        self.dataset = self.dataset[self.dataset['status'] == self.run_type].reset_index(drop=True)
        self.scans = []
        self.target = []
        self.labels = []
        self.positive_scans = []
        
        if self.run_type in ['train', 'test', 'valid']:
            if self.args.fast_dev_run == True:
                counter = self.args.fast_batch_size
            else:
                counter = len(self.dataset)
            
            self.max_classes = 2
            for index in tqdm(range(counter), desc=f"{self.run_type} data"):
                series, series_path, pos_index, neg_index, max_slice, labels = (self.dataset.loc[index,"series"], 
                                                                        self.dataset.loc[index,"series_path"], 
                                                                        self.dataset.loc[index,"pos_index"], 
                                                                        self.dataset.loc[index,"neg_index"],
                                                                        self.dataset.loc[index,"max_slice"],
                                                                        self.dataset.loc[index,"labels"])
                
                if self.args.only_annotated:
                    e = []
                    for key in labels:
                        e.extend(labels[key])
                    if len(e) < 1:
                        continue
                
                for i in range(1, max_slice-1):
                    if i not in pos_index:
                        if self.args.only_annotated:
                            continue
                    i_name = '0'*(3-len(str(i)))+str(i)
                    self.scans.append(os.path.join(series_path, f"slice_{i_name}.dcm"))
                    self.target.append(os.path.join(series_path, f"annot_{i_name}.dcm"))
                    self.labels.append(labels)

                    if i in pos_index:
                        self.positive_scans.append(len(self.scans)-1)
                    
        if self.args.mode == "multiclass":
            self.max_classes += 1
        
        print(f"{len(self.scans)} slices present in {self.run_type} data.\n")
    
    def initialize_transform(self, annot=False):
        if annot:
            interpolation = transforms.InterpolationMode.NEAREST
        else:
            interpolation = transforms.InterpolationMode.BILINEAR
            
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize(self.input_shape, interpolation)
                                  ])
    
    def window_channels(self):
        # Take stroke window instead of bone window(add as an arguments)
        # self.window_as_channel = windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window, windowing.blood_window, windowing.stroke_window])
        if self.args.windowing == "old":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window, 
                                                             windowing.blood_window, 
                                                             windowing.stroke_window])
        elif self.args.windowing == "brain":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window])

        elif self.args.windowing == "brain-infarcts":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window,
                                                                windowing.acute_stroke_window,
                                                                windowing.chronic_stroke_window])
        elif self.args.windowing == "gray-white-matter":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window,
                                                                windowing.acute_stroke_window,
                                                                windowing.chronic_stroke_window])

        elif self.args.windowing == "stroke-infarcts":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.stroke_window,
                                                                windowing.acute_stroke_window,
                                                                windowing.chronic_stroke_window])
        else:
            raise ValueError(f"Unknown windowing type: {self.args.windowing}")
            exit(0)
    
    def get_scan_channels(self, scan):
        index = self.get_slice_index(scan)
        index_1 = str(index-1)
        index_2 = str(index+1)
        index_1 = "0"*(3-len(index_1))+index_1
        index_2 = "0"*(3-len(index_2))+index_2
        series_path = self.get_series_path(scan)
        scan_a = windowing.brain_window(pydicom.read_file(scan).pixel_array)
        scan_b = windowing.brain_window(pydicom.read_file(os.path.join(series_path, f"slice_{index_1}.dcm")).pixel_array)
        scan_c = windowing.brain_window(pydicom.read_file(os.path.join(series_path, f"slice_{index_2}.dcm")).pixel_array)
        scan = np.stack([scan_a, scan_b, scan_c])
        return scan
            
        
    def get_foreground_crop(self, sitk_arr, annot_arr):
        cropped_sitk_arr, coord_top, coord_bottom = CropForeground(return_coords=True)(sitk_arr)
        cropped_annot_arr = annot_arr[:, coord_top[0]:coord_bottom[0], coord_top[1]:coord_bottom[1]]
        cropped_annot_arr[cropped_annot_arr>0] = 1
        return cropped_sitk_arr.numpy(), cropped_annot_arr.numpy()
    
    def move_channel_axis(self, scan, target):
        return np.moveaxis(scan,0,-1), np.moveaxis(target,0,-1)

    def preprocess(self, scan, target, crop):
        scan = self.get_scan_channels(scan)
        if crop:
            scan, target = self.get_foreground_crop(scan, target)
        scan, target = self.move_channel_axis(scan, target)
        scan = self.initialize_transform()(scan)
        target = self.initialize_transform(annot=True)(target)
        return scan, target
    
    def process(self, scan, target, crop=True):
        scan = self.preprocess(scan, target, crop)
        return scan
        
    def excute_augmentations(self, scan, target):
        if self.run_type == 'train' and self.args.augmentation:
            scan, target = self.train_transforms(scan, target)
        return scan, target

    def transform_labels(self, target, index):
        label_keys = list(self.labels[index].keys())
        annotation = pydicom.read_file(target).pixel_array

        if self.args.mode == "binary":
            annotation = np.where(np.isin(annotation, self.labels[index][label_keys[0]]), 1, 0)
        
        elif self.args.mode in [ "multilabel", "multiclass"]:
            if len(label_keys) < 2:
                raise ValueError(f"Use binary mode for training")
            
            annotations = []
            for i, key in enumerate(label_keys):
                if self.args.mode == "multilabel":
                    annotations.append(np.where(np.isin(annotation, self.labels[index][key]), 1, 0))
                elif self.args.mode == "multiclass":
                    annotations.append(np.where(np.isin(annotation, self.labels[index][key]), i+1, 0))

            if self.args.mode == "multilabel":
                annotation = np.stack(annotations)
            elif self.args.mode == "multiclass":
                annotation = np.maximum.reduce(annotations)
            
        return annotation
        
    
    def __getitem__(self, index):
        scan, target = self.scans[index], self.target[index]
        target = self.transform_labels(target, index)
        scan, target = self.process(scan, target, self.args.crop)
        return self.excute_augmentations(scan, target)

    def __len__(self):
        return int(len(self.scans))
    
    def train_transforms(self, input_scan, target):
        seed_ = random.random()
        state = torch.get_rng_state()
        if not self.args.extra_augs:
            if seed_ < 0.6:
                random_rotate = transforms.RandomAffine(scale=(0.9,1.3), degrees=(-45,45), interpolation=transforms.InterpolationMode.BILINEAR)
                if self.args.dataset_type == "2D":
                    input_scan, target = apply_torch_transform(random_rotate, state, input_scan, target)
                if self.args.dataset_type == "3D":
                    input_scan, target = apply_torch_transform_3d(random_rotate, state, self.spatial_dim_idx, input_scan, target)
            elif seed_ > 0.5:
                random_flip = transforms.RandomHorizontalFlip(p=1)
                if self.args.dataset_type == "2D":
                    input_scan, target = apply_torch_transform(random_rotate, state, input_scan, target)
                if self.args.dataset_type == "3D":
                    input_scan, target = apply_torch_transform_3d(random_flip, state, self.spatial_dim_idx, input_scan, target)
        
        elif self.args.extra_augs:
            # Random transforms
            if seed_ < 0.6:
                input_scan = transforms.GaussianBlur(5, sigma=(0.1, 2.0))(input_scan)
            
        return input_scan, target