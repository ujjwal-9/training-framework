import os
import re
import cv2
import json
import yaml
import munch
import shutil
import sqlite3
import pickle
import pydicom
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from glob import glob
from pprint import pprint
from pydicom import dcmread
from natsort import natsorted
from tqdm.notebook import tqdm
from collections import defaultdict

from skimage import io, color
from skimage.segmentation import mark_boundaries

import torch
import torch.nn.functional as F
import torchvision.transforms as tfms

from qure_series_classifier.predict import Infer
from qer.utils.postprocessing import mask_volume
from qer_utils.preprocessing import windowing, brain_window
from qer_utils.imageoperations import resampler, imread_3d

from ipywidgets import interact, widgets


import qer
from qer.predictor import get_predictions
qer.settings.checkpoints_path = "/home/users/ujjwal.upadhyay/packages/tests/qer/resources/checkpoints/"

from qer.predictor.utils.model_config import default_model_configs


import textwrap
from PIL import Image, ImageFont, ImageDraw, ImageOps
def create_report_slice(report, study_uid, series, series_dict, save_path, page_lines=40):
    report_lines = report.split('\n')
    n_lines = len(report_lines)
    report_slices = []
    save_dir = os.path.join(save_path, series)
    if not os.path.exists(save_path):
        os.mkdir(save_dir)
    filenames = [x["FilePath"] for x in series_dict["InstancesList"]]
    page = int(n_lines/page_lines)+1
    for i in range(page):
        report_slice = Image.new('L', (510, 716), color=0)
        report_slice = ImageOps.expand(report_slice, 1)
        draw = ImageDraw.Draw(report_slice)
        font = ImageFont.truetype("/home/users/arjun.agarwal/projects/qer_sc/src/qer_sc/data/AvenirLTStd-Book.otf", size=12)
        report_text = '\n'.join([textwrap.fill(line.strip(), width=85) for line in report_lines[i*page_lines:(i+1)*page_lines]])
        draw.text((10, 10), report_text, font=font, fill=1)
        report_slices.append(report_slice)
        save_file_path = f"{save_dir}/{study_uid}_{i}.bmp"
        out_dcm_path = save_file_path[:-len(".bmp")] + ".dcm"
        report_slice.save(save_file_path)
        if i == 0:
            first_path = out_dcm_path
            os.system(f"img2dcm -i BMP --study-from {filenames[0]} {save_file_path} {out_dcm_path}")
        else:
            os.system(f"img2dcm -i BMP --study-from {first_path} {save_file_path} {out_dcm_path}")
        
        first_dcm = pydicom.read_file(first_path)
        dcm_file = pydicom.read_file(out_dcm_path)
        
        dcm_file.InstanceNumber = page-i
        dcm_file.SeriesInstanceUID = first_dcm.SeriesInstanceUID
        
        dcm_file.pixel_array[dcm_file.pixel_array==0] = 0
        dcm_file.pixel_array[dcm_file.pixel_array==1] = 441.6
        
        dcm_file.save_as(out_dcm_path)
        os.system(f"rm {save_file_path}")
    return report_slices

def get_qer_config(query=None, model_name=None):
    if query is None:
        return ["model_names", "model_config", "all_config"]
    elif query == "model_names":
        model_names = []
        for config in default_model_configs.configs:
            model_names.append(config.name)
        return model_names
    elif query == "model_config":
        for config in default_model_configs.configs:
            if config.name == model_name:
                return config
    elif query == "all_config":
        all_configs = []
        for config in default_model_configs.configs:
            all_configs.append(config)
        return all_configs


def get_ct_batch_for_studyuid(filepath, sampling_dir="/data_nas2/processed/HeadCT/sampling/"):
        if filepath is None:
            return None
        if os.path.exists(os.path.join(sampling_dir, 'ct_batch_3', filepath)):
            return os.path.join(sampling_dir, 'ct_batch_3', filepath)
        elif os.path.exists(os.path.join(sampling_dir, 'ct_batches_all', filepath)):
            return os.path.join(sampling_dir, 'ct_batches_all', filepath)
        elif os.path.exists(os.path.join(sampling_dir, 'ct_batch_1', filepath)):
            return os.path.join(sampling_dir, 'ct_batch_1', filepath)
        elif os.path.exists(os.path.join(sampling_dir, 'ct_batch_2', filepath)):
            return os.path.join(sampling_dir, 'ct_batch_2', filepath)
        else:
            return None

def get_sqlite_db(sqlite_path="/home/users/ujjwal.upadhyay/packages/head-ct-annotations-database/database/data/all_studies.sqlite"):
    sql_query = "SELECT * FROM NLP"
    conn = sqlite3.connect(sqlite_path)
    df = pd.read_sql(sql_query, con=conn)
    return df
    
def get_col(name, df):
    return list(df.columns).index(name)
    
def infer_series_classifier(img_based=True, body_part="brain"):
    validation_conf = {
        "isc": {
            "use": img_based
        }
    }
    infer = Infer(body_part, validation_conf)
    return infer
    
    
def get_colored_heatmap(raw_heatmap, original_image):
    def _get_raw_heatmap(pixel_pred):
        cv2_img_size = original_image.shape[::-1]
        pixel_pred = np.uint8(pixel_pred * 255)
        pixel_pred = cv2.resize(pixel_pred, cv2_img_size)
        return pixel_pred


    def _gradient(start, end, num_steps):
        return np.linspace(start, end, num_steps)

    def _qure_color_map():
        b1 = [0, 110, 109, 0]
        b2 = [65, 232, 229, 180]
        b3 = [255, 255, 0, 200]
        b4 = [255, 0, 0, 220]

        final_cmap = np.concatenate(
            [_gradient(b1, b2, 80), _gradient(b2, b3, 80), _gradient(b3, b4, 80)]
        )
        return final_cmap

    raw_heatmap = _get_raw_heatmap(raw_heatmap)

    # smoothen the image
    kernel_frac = 0.05
    kernel_size = int(kernel_frac * max(raw_heatmap.shape))
    kernel_size = 2 * (kernel_size // 2) + 1
    heatmap_blurred = cv2.GaussianBlur(raw_heatmap, (kernel_size, kernel_size), 0)

    # color map
    final_cmap = _qure_color_map()
    num_steps = final_cmap.shape[0]
    heatmap_quantisied = np.uint8(heatmap_blurred / 255 * (num_steps - 1))
    heatmap_color = np.uint8(final_cmap[heatmap_quantisied])

    # alpha merge
    heatmap_alpha = heatmap_color[:, :, 3] / 255
    heatmap_rgb = heatmap_color[:, :, :3]

    colored_heatmap = np.uint8(
        heatmap_rgb * heatmap_alpha[:, :, np.newaxis]
        + original_image[:, :, np.newaxis] * (1 - heatmap_alpha)[:, :, np.newaxis]
    )

    return colored_heatmap


def remove_unnecessary_files(folder):
    filenames = glob(os.path.join(folder, "*"))
    to_remove = ["report.dcm", "results", "report.bmp", "summary.png", "gsps", "reports", "sc", "series_thumbnail.png", "sc_thumbnail", "aspects_sc", "thumbnails"]
    for file_to_remove in to_remove:
        rm = os.path.join(folder, file_to_remove)
        if rm in filenames:
            filenames.remove(rm)
    return filenames

def get_necessary_files(folder):
    filenames = glob(os.path.join(folder, "*"))
    component = '[a-z0-9]{8}'
    pattern = f'^.*{component}-{component}-{component}-{component}-{component}$'
    filepaths = []
    for filepath in filenames:
        if re.match(pattern, filepath):
            filepaths.append(filepath)
    return filepaths

def get_db():
    import qer_utils.db as db
    db = db.get_mongo_db()
    return db

def get_scan(series, db):
    series_dict = db.dicoms.find_one({"_id": series})
    filenames = [x["FilePath"] for x in series_dict["InstancesList"]]
    return resampler.load_sitk_image(filenames)


def read_sitk_from_dcms(filenames):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(filenames)
    image = reader.Execute()
    return image

def load_model(names):
    models = []
    for name in names:
        q_model = get_predictions.load_model(name)
        q_model.model = q_model.model.cuda()
        models.append(q_model)
    return models


def thresholding(pred, threshold=0.01):
    pred[pred > threshold] = 1
    pred[pred < threshold] = 0

    return pred.int()

def thresholding_numpy(pred, threshold=0.01):
    pred[pred > threshold] = 1
    pred[pred < threshold] = 0

    return pred.astype("int")

def get_arr(sitk_img):
    return sitk.GetArrayFromImage(sitk_img)

def read_series(series, path):
    file_names = natsorted(glob(f"{path}/images/*"))
    file_names = [name for name in file_names if not os.path.isdir(name)]
    valid_names = []
    for file in tqdm(file_names):
        series_uid = str(dcmread(file)["SeriesInstanceUID"]).split(" ")[-1]
        if series_uid == series:
            valid_names.append(file)
    return resampler.load_sitk_image(valid_names, resample=False), valid_names

def resize(size=(512,512), mode=tfms.InterpolationMode.NEAREST):
    return tfms.Resize(size, mode)


def plot_model_output(scan_arr, model_output, idx, threshold=0.7, ww=80, wl=40, color=True, figsize=(16,8)):
    window = windowing.window_generator(ww, wl)
    target_img = windowing.brain_window(scan_arr[idx])
    input_img = window(scan_arr[idx])
    acute_stroke = windowing.acute_stroke_window(scan_arr[idx])
    
    f = plt.figure(figsize=figsize)
    
    plt.subplot(131, title = "Input Slice")
    plt.imshow(input_img, 'gray', interpolation='none')
    
    plt.subplot(132, title = "Acute Stroke Window")
    plt.imshow(acute_stroke, 'gray', interpolation='none')

    if model_output is not None:
        plt.subplot(133, title = "Predicted Mask")
        if color:
            masked_img_out = color.label2rgb(model_output[idx], target_img, colors=[(255,0,0),(0,0,255)], alpha=0.01, bg_label=0, bg_color=None)
        else:
            masked_img_out = mark_boundaries(input_img, model_output[idx].astype("int"), color=(255,255,255))
        plt.imshow(masked_img_out)
    
    plt.show()
    
def plot_all_scan(scan_arr, model_output=None, threshold=0.7, color=True, figsize=(16,8)):
    total_slice = scan_arr.shape[0]
    threshold_dynamic = None
    print(threshold)
    if threshold == "NA":
        threshold_slider = threshold
    else:
        threshold_slider = widgets.FloatSlider(value=threshold, min=0, max=1, step=0.01)
        
    def callback(idx, threshold, ww, wl):
        model_out_fixed = None
        if model_output is not None:
            if threshold == "NA":
                model_out_fixed = model_output
                print("HERE")
            else:
                if threshold_dynamic is None or threshold_dynamic != threshold:
                    model_out_fixed = resize(torch.from_numpy(thresholding_numpy(model_output, threshold))).numpy()
                    print(f"INFARCTS: {np.argwhere(model_out_fixed.sum(axis=(1,2))>0)[:,0]}")
        plot_model_output(scan_arr, model_out_fixed, idx, threshold, ww, wl, color, figsize)
        
    interact(
        callback,
        idx = widgets.IntSlider(value=0, min=0, max=total_slice-1, step=1),
        threshold = threshold_slider,
        ww = widgets.IntSlider(value=80, min=0, max=600, step=1),
        wl = widgets.IntSlider(value=40, min=0, max=600, step=1)
    )

def plot_scan(scan_arr, do_windowing=True, figsize=(16,8)):
    total_slice = scan_arr.shape[0]
    def callback(idx, ww=40, wl=40):
        if do_windowing:
            window = windowing.window_generator(ww, wl)
        plt.figure(figsize=figsize)
        plt.axis("off")
        if do_windowing:
            plt.imshow(window(scan_arr[idx]), cmap="gray")
        else:
            plt.imshow(scan_arr[idx], cmap="gray")
        plt.show()
    if do_windowing:
        interact(
            callback,
            idx = widgets.IntSlider(value=int(total_slice/2), min=0, max=total_slice-1, step=1),
            ww = widgets.IntSlider(value=80, min=0, max=600, step=1),
            wl = widgets.IntSlider(value=40, min=0, max=600, step=1)
        )
    else:
        interact(
            callback,
            idx = widgets.IntSlider(value=int(total_slice/2), min=0, max=total_slice-1, step=1),
        )

        
def plot_side_by_side(scan_arrs, do_windowing=True, figsize=(16,8)):
    total_slice = scan_arrs[0].shape[0]
    def callback(idx, ww=40, wl=40):
        if do_windowing:
            window = windowing.window_generator(ww, wl)
        plt.figure(figsize=figsize)
        plt.axis("off")
        plt.subplot(1,2,1)
        if do_windowing:
            plt.imshow(window(scan_arrs[0][idx]), cmap="gray")
        else:
            plt.imshow(scan_arrs[0][idx], cmap="gray")
        plt.subplot(1,2,2)
        if do_windowing:
            plt.imshow(window(scan_arrs[1][idx]), cmap="gray")
        else:
            plt.imshow(scan_arrs[1][idx], cmap="gray")
        plt.show()
    if do_windowing:
        interact(
            callback,
            idx = widgets.IntSlider(value=int(total_slice/2), min=0, max=total_slice-1, step=1),
            ww = widgets.IntSlider(value=80, min=0, max=600, step=1),
            wl = widgets.IntSlider(value=40, min=0, max=600, step=1)
        )
    else:
        interact(
            callback,
            idx = widgets.IntSlider(value=int(total_slice/2), min=0, max=total_slice-1, step=1),
        )
        
def plot_aspects_output(scan_arrs, do_windowing=True, figsize=(16,8)):
    total_slice = scan_arrs[0].shape[0]
    def callback(idx, ww=40, wl=80):
        if do_windowing:
            window = windowing.window_generator(ww, wl)
        plt.figure(figsize=figsize)
        plt.axis("off")
        
        plt.subplot(1,2,1)
        if do_windowing:
            windowed_scan = window(scan_arrs[0][idx])
            plt.imshow(windowed_scan, cmap="gray")
        else:
            plt.imshow(scan_arrs[0][idx], cmap="gray")
        
        plt.subplot(1,2,2)
        if do_windowing:
            out = mark_boundaries(windowed_scan, scan_arrs[1][idx].astype("int"), color=(255,255,255))
            plt.imshow(out, cmap="gray")
        else:
            out = mark_boundaries(scan_arrs[0][idx], scan_arrs[1][idx].astype("int"), color=(255,255,255))
            plt.imshow(scan_arrs[1][idx], cmap="gray")
        plt.show()
    
    if do_windowing:
        interact(
            callback,
            idx = widgets.IntSlider(value=int(total_slice/2), min=0, max=total_slice-1, step=1),
            ww = widgets.IntSlider(value=80, min=0, max=600, step=1),
            wl = widgets.IntSlider(value=40, min=0, max=600, step=1)
        )
    else:
        interact(
            callback,
            idx = widgets.IntSlider(value=int(total_slice/2), min=0, max=total_slice-1, step=1),
        )
        
def plot_infarcts_output(scan_arrs, do_windowing=True, figsize=(16,8)):
    total_slice = scan_arrs[0].shape[0]
    def callback(idx, ww=40, wl=80):
        if do_windowing:
            window = windowing.window_generator(ww, wl)
        plt.figure(figsize=figsize)
        plt.axis("off")
        
        plt.subplot(1,3,1)
        if do_windowing:
            windowed_scan = window(scan_arrs[0][idx])
            plt.imshow(windowed_scan, cmap="gray")
        else:
            plt.imshow(scan_arrs[0][idx], cmap="gray")
        
        plt.subplot(1,3,2)
        if do_windowing:
            out = color.label2rgb(scan_arrs[1][idx]==1, windowed_scan, colors=[(255,0,0),(0,0,255)], alpha=0.01, bg_label=0, bg_color=None)
            plt.imshow(out, cmap="gray")
        else:
            out = color.label2rgb(scan_arrs[1][idx], windowed_scan, colors=[(255,0,0),(0,0,255)], alpha=0.01, bg_label=0, bg_color=None)
            plt.imshow(scan_arrs[1][idx], cmap="gray")
        
        plt.subplot(1,3,3)
        if do_windowing:
            out = color.label2rgb(scan_arrs[1][idx]==2, windowed_scan, colors=[(255,0,0),(0,0,255)], alpha=0.01, bg_label=0, bg_color=None)
            plt.imshow(out, cmap="gray")
        else:
            out = color.label2rgb(scan_arrs[1][idx], windowed_scan, colors=[(255,0,0),(0,0,255)], alpha=0.01, bg_label=0, bg_color=None)
            plt.imshow(scan_arrs[1][idx], cmap="gray")
        plt.show()
    
    if do_windowing:
        interact(
            callback,
            idx = widgets.IntSlider(value=int(total_slice/2), min=0, max=total_slice-1, step=1),
            ww = widgets.IntSlider(value=80, min=0, max=600, step=1),
            wl = widgets.IntSlider(value=40, min=0, max=600, step=1)
        )
    else:
        interact(
            callback,
            idx = widgets.IntSlider(value=int(total_slice/2), min=0, max=total_slice-1, step=1),
            label = widgets.IntSlider(value=1, min=0, max=2, step=1)
        )
        
def plot_two_aspects_output(scan_arrs, do_windowing=True, figsize=(16,8)):
    total_slice = scan_arrs[0].shape[0]
    def callback(idx, ww=40, wl=80):
        if do_windowing:
            window = windowing.window_generator(ww, wl)
            windowed_scan = window(scan_arrs[0][idx])
        
        plt.figure(figsize=figsize)
        plt.axis("off")
        
        plt.subplot(1,2,1)
        if do_windowing:
            out_1 = mark_boundaries(windowed_scan, scan_arrs[1][idx].astype("int"), color=(255,255,255))
            plt.imshow(out_1, cmap="gray")
        else:
            out = mark_boundaries(scan_arrs[0][idx], scan_arrs[1][idx].astype("int"), color=(255,255,255))
            plt.imshow(scan_arrs[1][idx], cmap="gray")
        
        plt.subplot(1,2,2)
        if do_windowing:
            out_2 = mark_boundaries(windowed_scan, scan_arrs[2][idx].astype("int"), color=(255,255,255))
            plt.imshow(out_2, cmap="gray")
        else:
            out_2 = mark_boundaries(scan_arrs[0][idx], scan_arrs[2][idx].astype("int"), color=(255,255,255))
            plt.imshow(scan_arrs[2][idx], cmap="gray")
        plt.show()
    
    if do_windowing:
        interact(
            callback,
            idx = widgets.IntSlider(value=int(total_slice/2), min=0, max=total_slice-1, step=1),
            ww = widgets.IntSlider(value=80, min=0, max=600, step=1),
            wl = widgets.IntSlider(value=40, min=0, max=600, step=1)
        )
    else:
        interact(
            callback,
            idx = widgets.IntSlider(value=int(total_slice/2), min=0, max=total_slice-1, step=1),
        )


        
def plot_aspects_and_infarcts(scan_arrs, do_windowing=True, figsize=(16,8)):
    total_slice = scan_arrs[0].shape[0]
    def callback(idx, ww=40, wl=80):
        if do_windowing:
            window = windowing.window_generator(ww, wl)
        plt.figure(figsize=figsize)
        plt.axis("off")
        
        plt.subplot(1,2,1)
        if do_windowing:
            windowed_scan = window(scan_arrs[0][idx])
            plt.imshow(windowed_scan, cmap="gray")
        else:
            plt.imshow(scan_arrs[0][idx], cmap="gray")
        
        plt.subplot(1,2,2)
        if do_windowing:
            out = mark_boundaries(windowed_scan, scan_arrs[1][idx].astype("int"), color=(255,255,255))
            masked_img_out = color.label2rgb(scan_arrs[2][idx], out, colors=[(255,0,0),(0,0,255)], alpha=0.01, bg_label=0, bg_color=None)
            plt.imshow(masked_img_out, cmap="gray")
        else:
            out = mark_boundaries(scan_arrs[0][idx], scan_arrs[1][idx].astype("int"), color=(255,255,255))
            masked_img_out = color.label2rgb(scan_arrs[2][idx], out, colors=[(255,0,0),(0,0,255)], alpha=0.01, bg_label=0, bg_color=None)
            plt.imshow(masked_img_out, cmap="gray")
        plt.show()
    
    if do_windowing:
        interact(
            callback,
            idx = widgets.IntSlider(value=int(total_slice/2), min=0, max=total_slice-1, step=1),
            ww = widgets.IntSlider(value=80, min=0, max=600, step=1),
            wl = widgets.IntSlider(value=40, min=0, max=600, step=1)
        )
    else:
        interact(
            callback,
            idx = widgets.IntSlider(value=int(total_slice/2), min=0, max=total_slice-1, step=1),
        )
        
def plot_screens(scan_arrs, do_windowing, overlay, title, org_scan_idx=0, figsize=(16,8)):
    total_slice = scan_arrs[0].shape[0]
    total_screens = len(scan_arrs)
    
    def windowing_scan(scan, ww, wl, window=False):
        if window:
            return windowing.window_generator(ww, wl)(scan)
        return scan
    
    def callback(idx, ww=40, wl=40, toggle=0):
        plt.figure(figsize=figsize)
        for i in range(total_screens):
            plt.subplot(1,total_screens,i+1)
            plt.axis("off")
            plt.title(title[i])
            to_plot = windowing_scan(scan_arrs[org_scan_idx][idx], ww, wl, do_windowing[i])
            if not toggle and overlay[i]:
                to_plot = color.label2rgb(scan_arrs[i][idx], to_plot, colors=[(255,0,0),(0,0,255)], alpha=0.01, bg_label=0, bg_color=None)
            plt.imshow(to_plot, cmap="gray")
        plt.show()
    
    if do_windowing:
        interact(
            callback,
            idx = widgets.IntSlider(value=int(total_slice/2), min=0, max=total_slice-1, step=1),
            ww = widgets.IntSlider(value=80, min=0, max=600, step=1),
            wl = widgets.IntSlider(value=40, min=0, max=600, step=1),
            toggle=widgets.IntSlider(value=0, min=0, max=1, step=1)
        )
    else:
        interact(
            callback,
            idx = widgets.IntSlider(value=int(total_slice/2), min=0, max=total_slice-1, step=1),
        )