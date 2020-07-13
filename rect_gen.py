import cv2
import numpy as np
from random import randrange


def generate():
    size = (64, 64, 3)
    color = (255, 255, 255)
    blank_image = np.zeros(size, np.uint8)
    mi, ma = 20, 44
    p1 = (randrange(mi), randrange(mi))
    p2 = (randrange(mi) + ma, randrange(mi))
    p3 = (randrange(mi) + ma, randrange(mi) + ma)
    p4 = (randrange(mi), randrange(mi) + ma)

    cv2.line(blank_image, p1, p2, color, 1)
    cv2.line(blank_image, p2, p3, color, 1)
    cv2.line(blank_image, p3, p4, color, 1)
    cv2.line(blank_image, p4, p1, color, 1)
    return blank_image, (p1, p2, p3, p4)


i = 0
while i < 1024:
    img, cord = generate()
    file = "rect/train/images/{0}.jpg".format(i)
    cv2.imwrite(file, img)
    with open("rect/train/labels/{0}.txt".format(i), 'w') as f1:
        f1.write("{0} {1} {2} {3} {4} {5} {6} {7}".format(
            cord[0][0] / 64, cord[0][1] / 64,
            cord[1][0] / 64, cord[1][1] / 64,
            cord[2][0] / 64, cord[2][1] / 64,
            cord[3][0] / 64, cord[3][1] / 64,
        ))
    i += 1
