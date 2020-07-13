from rect_gen import generate
import onnxruntime
import numpy as np
import cv2
from matplotlib import pyplot as plt

ort_session = onnxruntime.InferenceSession("rect.onnx")


def detect_file(path):
    raw = cv2.imread(path, cv2.IMREAD_COLOR)
    raw = cv2.resize(raw, (64, 64))
    edges = cv2.Canny(raw, 100, 500)
    img = cv2.merge([edges, edges, edges])
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5])
    val = np.array([0.5, 0.5, 0.5])
    img = (img - mean) / val
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    output = ort_session.run(None, {ort_session.get_inputs()[0].name: img})
    coor = (output[0][0]) * 64
    print(coor)
    red = (0, 0, 255)
    cv2.circle(raw, (coor[0], coor[1]), 2, red, 2)
    cv2.circle(raw, (coor[2], coor[3]), 2, red, 2)
    cv2.circle(raw, (coor[4], coor[5]), 2, red, 2)
    cv2.circle(raw, (coor[6], coor[7]), 2, red, 2)
    return raw


def detect_generate():
    raw, cor = generate()
    img = np.array(raw)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5])
    val = np.array([0.5, 0.5, 0.5])
    img = (img - mean) / val
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    output = ort_session.run(None, {ort_session.get_inputs()[0].name: img})
    coor = (output[0][0]) * 64
    print(coor)
    red = (0, 0, 255)
    cv2.circle(raw, (coor[0], coor[1]), 2, red, 2)
    cv2.circle(raw, (coor[2], coor[3]), 2, red, 2)
    cv2.circle(raw, (coor[4], coor[5]), 2, red, 2)
    cv2.circle(raw, (coor[6], coor[7]), 2, red, 2)
    return raw


img = detect_file("c1.jpg")
plt.imshow(img)
plt.show()

# img = cv2.imread('c1.jpg', 0)
# edges = cv2.Canny(img, 100, 500)
