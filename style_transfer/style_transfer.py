#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import functools
import time

# Third Party Libraries
import numpy as np
import PIL.Image
import tensorflow as tf
import tensorflow_hub as hub


def tensor_to_image(tensor):

    """Convert a Tensor output to a Image using PIl

    Args:
        tensor (int):

    Return:
        image (PIL Image)

    """

    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
        img = PIL.Image.fromarray(tensor)
    return img


def load_img(path_to_img):

    """Load an image for the style model

    Args:
        path_to_img (String): Path to the image you want to load

    Return:
        img (Image): The image uploaded as a tensorflow image

    """

    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


if __name__ == "__main__":

    content_path = "/home/kmurakami/git/style-transfer-tensorflow/input_image.jpg"
    style_path = "/home/kmurakami/git/style-transfer-tensorflow/style_image.jpg"
    
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    tensor_to_image(stylized_image).save("stylized-image.png")
    print("Finished")