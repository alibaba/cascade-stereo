import numpy as np
import re
import sys


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()

import random, cv2
class RandomCrop(object):
    def __init__(self, CropSize=0.1):
        self.CropSize = CropSize

    def __call__(self, image, normal):
        h, w = normal.shape[:2]
        img_h, img_w = image.shape[:2]
        CropSize_w, CropSize_h = max(1, int(w * self.CropSize)), max(1, int(h * self.CropSize))
        x1, y1 = random.randint(0, CropSize_w), random.randint(0, CropSize_h)
        x2, y2 = random.randint(w - CropSize_w, w), random.randint(h - CropSize_h, h)

        normal_crop = normal[y1:y2, x1:x2]
        normal_resize = cv2.resize(normal_crop, (w, h), interpolation=cv2.INTER_NEAREST)

        image_crop = image[4*y1:4*y2, 4*x1:4*x2]
        image_resize = cv2.resize(image_crop, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        # import matplotlib.pyplot as plt
        # plt.subplot(2, 3, 1)
        # plt.imshow(image)
        # plt.subplot(2, 3, 2)
        # plt.imshow(image_crop)
        # plt.subplot(2, 3, 3)
        # plt.imshow(image_resize)
        #
        # plt.subplot(2, 3, 4)
        # plt.imshow((normal + 1.0) / 2, cmap="rainbow")
        # plt.subplot(2, 3, 5)
        # plt.imshow((normal_crop + 1.0) / 2, cmap="rainbow")
        # plt.subplot(2, 3, 6)
        # plt.imshow((normal_resize + 1.0) / 2, cmap="rainbow")
        # plt.show()
        # plt.pause(1)
        # plt.close()

        return image_resize, normal_resize