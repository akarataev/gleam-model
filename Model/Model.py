import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image

from Utility import get_nb_files, get_data, plot_training, resize_image, plot_preds


if __name__ == "__main__":

    a = argparse.ArgumentParser()

    a.add_argument("--train_dir", default='Dataset/Train')
    a.add_argument("--val_dir", default='Dataset/Validation')
    a.add_argument("--tl_epoch", default=15)
    a.add_argument("--ft_epoch", default=5)
    a.add_argument("--batch_size", default=30)
    a.add_argument("--output_model_file", default="vgg16.h5")
    a.add_argument("--image", help="path to image")
    a.add_argument("--ft_model", default="Models/ft_vl_vgg16.h5")
    a.add_argument("--tl_model", default="Models/tl_vl_vgg16.h5")
    a.add_argument("--featurewise_center", default=False)

    args = a.parse_args()
