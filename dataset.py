import os
import random
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

import networks
import loss

defect_free_folder = "./data/hazelnut/train/good/"
defect_folder = "./data/hazelnut/test/hole/"
mask_folder = "./data/hazelnut/ground_truth/hole/"

def load_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels= 3)
    return image

def mapping(img_path):
    image = load_image(img_path)
    return {"images": tf.cast(image, tf.float32)/255.0}

def load_dataset(folder, resolution):
    print("=====> Loading dataset at : " + os.path.abspath(folder))
    files = os.listdir(folder)
    files.sort()

    ds = []
    for file in files:
        try:
            tmp = Image.open(folder + file).convert('RGB').resize((resolution, resolution))
        except:
            print("Error")
            break
        tmp = np.array(tmp)
        ds.append(tmp)

    return np.array(ds)

def get_batch(dataset, batch_size):
    ids = np.random.randint(0, dataset.shape[0] - 1, batch_size)
    out = []

    for i in ids:
        out.append(dataset[i])
        if random.random() < 0.5:
            out[-1] = np.flip(out[-1], 1)

    return tf.convert_to_tensor(out, dtype= tf.float32)