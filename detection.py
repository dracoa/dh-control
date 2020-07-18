import onnxruntime
import numpy as np
from PIL import Image
import os

label = ['active', 'not_ready', 'ready']

ort_session = onnxruntime.InferenceSession("skill.onnx")


def is_ready(img):
    result = is_ready_np(img)
    x = result[0][0]
    smax = np.exp(x) / sum(np.exp(x))
    return np.argmax(smax) == 2


def is_ready_np(img):
    img = img.resize((56, 56), Image.BILINEAR)
    img = np.array(img)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5])
    val = np.array([0.5, 0.5, 0.5])
    img = (img - mean) / val
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return ort_session.run(None, {ort_session.get_inputs()[0].name: img})

