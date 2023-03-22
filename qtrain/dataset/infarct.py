import os
import json
import PIL
import glob
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
        if self.args.dataset_type == "3D":
            dataset = InfarctDataset3D(self.args, status)
        else:
            raise ValueError(f"Not Implemented")
        return dataset

    def setup_dataloader(self, dataset, batch_size):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle= True
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
    
#     def window_channels(self):
#         for window_config in self.args.windowing:
#             windows.append(windowing.window_generator(window_config))
            
#         return windowing.WindowsAsChannelsTransform(windows=windows)
    
    def window_channels(self):
        # Take stroke window instead of bone window(add as an arguments)
        # self.window_as_channel = windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window, windowing.blood_window, windowing.stroke_window])
        if self.args.windowing == "old":
            return windowing.WindowsAsChannelsTransform(windows=[windowing.brain_window, 
                                                             windowing.blood_window, 
                                                             windowing.stroke_window])
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
        scan_windowed = torch.cat(scan_windowed, dim=0).permute(3,1,2,0)
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
                    input_scan, target = apply_torch_transform_3d(random_rotate, state, 3, input_scan, target)
            elif seed_ > 0.5:
                random_flip = transforms.RandomHorizontalFlip(p=1)
                if self.args.dataset_type == "3D":
                    input_scan, target = apply_torch_transform_3d(random_flip, state, 3, input_scan, target)
        
        elif self.args.extra_augs:
            # Random transforms
            if seed_ < 0.6:
                input_scan = transforms.GaussianBlur(5, sigma=(0.1, 2.0))(input_scan)
            
        return input_scan, target
    
    