# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
from tensorflow.keras.applications.vgg16 import preprocess_input
from os import listdir
from os import path
import imghdr

pathLoadImg = 'dataGenerated/original/'
pathToSaveImg = 'dataGenerated/generated/dni'

def imageGenerator():
    print("loading example images ...")
    directory = pathLoadImg
    for name in listdir(directory):
        filename = path.join(directory, name)
        image = load_img(filename)

        image.mode  # Comprobamos el modo de la imagen: P
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image_id = name.split('.')[0]

        aug = ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")
        total = 0

        print(" generating images...")
        imageGen = aug.flow(image, batch_size=1, save_to_dir= pathToSaveImg,
                            save_prefix="image", save_format="jpg")

        for image in imageGen:

            total += 1

            if total == 100:
                break

imageGenerator()