import os
import yaml
import json
import PIL
import glob
import pydicom
import random
import platform
import h5py as h5
import numpy as np
import pandas as pd
import SimpleITK as sitk


import torch
import pytorch_lightning as pl

from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from warnings import warn
from safetensors import safe_open

from qtrain.utils.transforms import (
    RandomCrop,
    hu_windowing,
)


import structlog

logger = structlog.getLogger()


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

        self.seed = self.args.seed
        self.num_workers = self.args.num_workers
        self.batch_size = self.args.batch_size
        self.valid_batch_size = self.args.valid_batch_size

    def num_classes(self):
        return self.args.num_classes

    def get_num_training_samples(self):
        self.train_dataloader()
        return self.num_samples

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """Split the train, valid and test dataset"""
        pass

    def setup_dataset(self, status):
        if self.args.dataset_class_60k == True:
            dataset = InfarctDataset3D_60k(self.args, status)
        else:
            dataset = InfarctDataset3D_Fast(self.args, status)
        return dataset

    def setup_dataloader(self, dataset, batch_size, n_samples):
        sampler = None
        shuffle = True
        if self.args.sampler == True:
            if n_samples > 0:
                shuffle = None
                from torch.utils.data import WeightedRandomSampler

                sampler = WeightedRandomSampler(
                    dataset.dataset.sample_wts.to_list(), n_samples
                )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            drop_last=self.args.dl_drop_last,
            pin_memory=self.args.dl_pin_memory,
            shuffle=shuffle,
            sampler=sampler,
            prefetch_factor=self.args.prefetch,
        )
        return loader

    def train_dataloader(self):
        train_dataset = self.setup_dataset("train")
        self.num_classes = train_dataset.max_classes
        loader = self.setup_dataloader(
            train_dataset,
            self.batch_size,
            self.args.train_samples,
        )
        self.num_samples = len(loader)
        return loader

    def val_dataloader(self):
        loader = self.setup_dataloader(
            self.setup_dataset("valid"),
            self.valid_batch_size,
            self.args.valid_samples,
        )
        return loader

    # def val_dataloader(self):
    #     pass

    def test_dataloader(self):
        pass


class InfarctDataset3D_60k(Dataset):
    def __init__(self, args, run_type="train", init_dataset=True):
        self.args = args
        self.run_type = run_type
        self.input_shape = [self.args.img_size, self.args.img_size]
        self.window_as_channel = self.window_channels()
        if init_dataset:
            self.initialize_dataset(self.args.datapath)

        if self.run_type == "valid" or self.run_type == "test":
            self.args.n_slices = self.args.valid_n_slices

    def get_studyuid(self, index):
        return self.series[index]

    def append_dataset(
        self, scan, annotation, label, crop, cls, acute, chronic, normal, series
    ):
        self.datapath.append(scan)
        self.targetpath.append(annotation)
        self.labels.append(label)
        if self.args.crop:
            self.crops.append(np.array(eval(crop)))
        else:
            self.crops.append(None)
        self.class_annot.append(cls)
        self.acute.append(acute)
        self.chronic.append(chronic)
        self.normal.append(normal)
        self.series.append(series)

    def initialize_dataset(self, datapath):
        self.dataset = pd.read_json(datapath)

        self.dataset = self.dataset[
            self.dataset["status"] == self.run_type
        ].reset_index(drop=True)
        self.dataset.annotpath = self.dataset.annotpath.values.astype("str")

        self.datapath = []
        self.targetpath = []
        self.labels = []
        self.crops = []
        self.class_annot = []
        self.acute = []
        self.chronic = []
        self.normal = []
        self.series = []

        if self.run_type in ["train", "test", "valid"]:
            counter = len(self.dataset)

            self.max_classes = 0
            for index in tqdm(range(counter), desc=f"{self.run_type} data"):
                series = self.dataset.loc[index, "series"]
                filepath, annotpath, labels, crop, cls = (
                    self.dataset.loc[index, "filepath"],
                    self.dataset.loc[index, "annotpath"],
                    self.dataset.loc[index, "labels"],
                    self.dataset.loc[index, "crop"],
                    self.dataset.loc[index, "annotation"],
                )
                acute, chronic, normal = (
                    self.dataset.loc[index, "acute"],
                    self.dataset.loc[index, "chronic"],
                    self.dataset.loc[index, "normal"],
                )
                n_classes = len(labels)
                if n_classes > self.max_classes:
                    self.max_classes = n_classes
                if self.args.only_annotated:
                    if_annotated = False
                    for key in labels:
                        if n_classes > 0:
                            if_annotated = True
                    if if_annotated:
                        self.append_dataset(
                            filepath,
                            annotpath,
                            labels,
                            crop,
                            cls,
                            acute,
                            chronic,
                            normal,
                            series,
                        )
                elif self.args.only_annotated == False:
                    self.append_dataset(
                        filepath,
                        annotpath,
                        labels,
                        crop,
                        cls,
                        acute,
                        chronic,
                        normal,
                        series,
                    )
        if self.args.mode == "multiclass":
            self.max_classes += 1
        print(
            f"{len(self.datapath)} scans present in {self.run_type} data belonging to {self.max_classes} classes\n"
        )

    def window_channels(self):
        if self.args.windowing == "conv":
            from qer_utils.nn.windowing import get_windower, default_window_opts

            default_window_opts.intensity_augmnetation = False
            default_window_opts.window_inits = [(80, 40), (175, 50), (40, 40)]
            return get_windower(default_window_opts)

        elif self.args.windowing == "tensor_based":
            window_configs = torch.tensor([(80, 40), (175, 50), (40, 40)])
            return hu_windowing(window_configs)

        else:
            raise ValueError(f"Unknown windowing type: {self.args.windowing}")
            exit(0)

    def get_window_channels(self, scan):
        if (self.args.windowing == "conv") or (self.args.windowing == "tensor_based"):
            return self.window_as_channel(scan)
        else:
            scan_windowed = [
                torch.tensor(self.window_as_channel(scan[i].numpy())).unsqueeze(0)
                for i in range(scan.shape[0])
            ]
            scan_windowed = torch.cat(scan_windowed, dim=0)
            if scan_windowed.shape[3] == 1:
                scan_windowed = scan_windowed.permute(3, 0, 1, 2)
                self.spatial_dim_idx = 1
            else:
                scan_windowed = scan_windowed.permute(3, 1, 2, 0)
                self.spatial_dim_idx = 3
            return scan_windowed

    def get_foreground_crop(self, sitk_arr, annot_arr, index):
        crops = self.crops[index]
        if crops.shape[0] == 3:
            crops = crops[1:]
        cropped_sitk_arr = sitk_arr[
            :, crops[0, 0] : crops[0, 1], crops[1, 0] : crops[1, 1]
        ]
        cropped_annot_arr = annot_arr[
            :, crops[0, 0] : crops[0, 1], crops[1, 0] : crops[1, 1]
        ]
        return cropped_sitk_arr, cropped_annot_arr

    def process(self, index):
        annotation = self.process_annotation_label(index)
        if (
            self.datapath[index].split(".")[-1] == "safetensors"
            or self.datapath[index].split(".")[-1] == "safetensor"
        ):
            with safe_open(self.datapath[index], framework="pt", device="cpu") as f:
                for k in f.keys():
                    ct_scan = f.get_tensor(k).numpy()
        elif self.datapath[index].split(".")[-1] == "h5":
            ct_scan = h5.File(self.datapath[index], "r+")["arr"][:]
        elif self.datapath[index].split(".")[-1] == "npy":
            ct_scan = np.load(self.datapath[index])
        else:
            ct_scan = sitk.GetArrayFromImage(sitk.ReadImage(self.datapath[index]))

        if annotation is None:
            annotation = np.zeros_like(ct_scan)

        is_ignore_index = False
        if annotation.sum() == 0 and self.class_annot[index] == 1:
            annotation = np.ones_like(ct_scan) * -100
            is_ignore_index = True

        annotation[annotation > 0] = 1
        annotation = annotation.astype(np.int16)

        if self.args.crop:
            ct_scan, annotation = self.get_foreground_crop(ct_scan, annotation, index)

        ct_scan, annotation = torch.tensor(ct_scan), torch.tensor(annotation).to(
            torch.int16
        )

        if self.args.n_slices > 0:
            if ct_scan.shape[0] < self.args.n_slices:
                ct_scan = torch.cat(
                    [
                        ct_scan,
                        torch.zeros(
                            self.args.n_slices - ct_scan.shape[0], *ct_scan.shape[1:]
                        ).to(torch.int16),
                    ]
                )
                if is_ignore_index:
                    annotation = torch.cat(
                        [
                            annotation,
                            torch.ones(
                                self.args.n_slices - annotation.shape[0],
                                *annotation.shape[1:],
                            ).to(torch.int16)
                            * -100,
                        ]
                    )
                else:
                    annotation = torch.cat(
                        [
                            annotation,
                            torch.zeros(
                                self.args.n_slices - annotation.shape[0],
                                *annotation.shape[1:],
                            ).to(torch.int16),
                        ]
                    )
            elif ct_scan.shape[0] > self.args.n_slices:
                start_slice = np.random.randint(
                    0, ct_scan.shape[0] - self.args.n_slices
                )
                ct_scan = ct_scan[start_slice : start_slice + self.args.n_slices]
                annotation = annotation[start_slice : start_slice + self.args.n_slices]

        ct_scan = self.get_window_channels(ct_scan.to(torch.float))
        return (
            ct_scan,
            annotation,
            [1 - self.normal[index]],
            [self.acute[index], self.chronic[index]],
        )

    def process_annotation_label(self, index):
        label_keys = list(self.labels[index].keys())
        if self.targetpath[index] == "None":
            annotation = None
            return annotation
        elif (
            self.targetpath[index].split(".")[-1] == "safetensors"
            or self.targetpath[index].split(".")[-1] == "safetensor"
        ):
            with safe_open(self.targetpath[index], framework="pt", device="cpu") as f:
                for k in f.keys():
                    annotation = f.get_tensor(k).numpy()
        elif self.datapath[index].split(".")[-1] == "h5":
            annotation = h5.File(self.targetpath[index], "r+")["arr"][:]
        elif self.datapath[index].split(".")[-1] == "gz":
            annotation = sitk.GetArrayFromImage(sitk.ReadImage(self.targetpath[index]))

        if self.args.mode == "binary":
            annotation = np.where(
                np.isin(annotation, self.labels[index][label_keys[0]]), 1, 0
            )

        elif self.args.mode in ["multilabel", "multiclass"]:
            if len(label_keys) < 2:
                raise ValueError(f"Use binary mode for training")

            annotations = []
            for i, key in enumerate(label_keys):
                if self.args.mode == "multilabel":
                    annotations.append(
                        np.where(np.isin(annotation, self.labels[index][key]), 1, 0)
                    )
                elif self.args.mode == "multiclass":
                    annotations.append(
                        np.where(np.isin(annotation, self.labels[index][key]), i + 1, 0)
                    )

            if self.args.mode == "multilabel":
                annotation = np.stack(annotations)
            elif self.args.mode == "multiclass":
                annotation = np.maximum.reduce(annotations)

        return annotation

    def excute_augmentations(self, ct_scan, annotation):
        ct_scan, annotation = self.train_transforms(ct_scan, annotation)
        return ct_scan, annotation

    def __getitem__(self, index):
        try:
            ct_scan, annotation, label_class, infarct_type = self.process(index)
            return (
                *self.excute_augmentations(ct_scan, annotation),
                torch.Tensor(label_class).to(torch.int16),
                torch.Tensor(infarct_type).to(torch.int16),
                self.series[index],
            )
        except:
            ct_scan = torch.zeros(
                (self.args.n_slices, 3, self.args.img_size, self.args.img_size)
            )
            annotation = (
                torch.ones((self.args.n_slices, self.args.img_size, self.args.img_size))
                * -100
            )
            label_class = [0]
            infarct_type = [-100, -100]
            logger.debug(f'Issue with this file {self.datapath[index].split("/")[-1]}')
            return (
                ct_scan.to(torch.float),
                annotation.to(torch.int16),
                torch.Tensor(label_class).to(torch.int16),
                torch.Tensor(infarct_type).to(torch.int16),
                self.series[index],
            )

    def __len__(self):
        return len(self.datapath)

    def train_transforms(self, input_scan, target):
        if self.run_type == "train" and self.args.augmentation:
            state = torch.get_rng_state()
            if random.random() < 0.7:
                translate = [random.uniform(-2, 2), random.uniform(-2, 2)]
                scale = random.uniform(0.8, 1.2)
                angle = random.uniform(-10, 10)
                input_scan = TF.affine(
                    input_scan,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=0.0,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
                target = TF.affine(
                    target,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=0.0,
                    interpolation=transforms.InterpolationMode.NEAREST,
                )

            if random.random() > 0.5:
                input_scan = TF.hflip(input_scan)
                target = TF.hflip(target)

            if random.random() > 0.5:
                input_scan = torch.flip(input_scan, dims=(0,))
                target = torch.flip(target, dims=(0,))

            if random.random() < 0.6:
                input_scan = transforms.GaussianBlur(5, sigma=(0.1, 2.0))(input_scan)

            if random.random() < 0.4:
                input_scan = transforms.ColorJitter(
                    brightness=0.5, contrast=0.6, saturation=0.7, hue=0.3
                )(input_scan)

            if random.random() < 0.6:
                input_scan = transforms.RandomAdjustSharpness(
                    sharpness_factor=0.6, p=1
                )(input_scan)

            if random.random() < 0.1:
                input_scan = transforms.RandomSolarize(threshold=0.85, p=1)(input_scan)

            if random.random() < 0.4:
                input_scan, target = RandomCrop((300, 300))(
                    {"image": input_scan, "mask": target}
                )

        input_scan = transforms.Resize(
            self.input_shape, transforms.InterpolationMode.BILINEAR
        )(input_scan)
        target = transforms.Resize(
            self.input_shape, transforms.InterpolationMode.NEAREST
        )(target)

        return input_scan, target
