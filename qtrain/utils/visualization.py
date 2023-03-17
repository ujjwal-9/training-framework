import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import color

def plot_masked_image(input_image, input_mask, output_mask, save_image=False, image_path='../assets/image.png'):
    '''
    Return masked image along with the input image, input image with output mask, input image with input mask
    '''
    fig = plt.figure(figsize=(24,8))
    plt.subplot(131, title = "Input Slice")
    plt.imshow(input_image[0].numpy(), 'gray', interpolation='none')

    plt.subplot(132, title = "Ouput Mask")
    masked_img_out = color.label2rgb(output_mask.detach().numpy(), input_image[0].numpy(),colors=[(255,0,0),(0,0,255)], alpha=0.01, bg_label=0, bg_color=None)
    plt.imshow(masked_img_out)

    plt.subplot(133, title = "Input Mask")
    masked_img_in = color.label2rgb(input_mask.detach().numpy(), input_image[0].numpy(),colors=[(255,0,0),(0,0,255)], alpha=0.01, bg_label=0, bg_color=None)
    plt.imshow(masked_img_in)
    if save_image:
        fig.savefig(image_path)
    return fig
    