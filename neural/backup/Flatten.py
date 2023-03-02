from PIL import Image
import cv2
import numpy as np


def flatten(data, flat: int = (5, 5)):
    if isinstance(flat, int):
        flat = (flat, flat)

    if isinstance(data, str):
        im = Image.open(data)

    elif isinstance(data, (list, np.ndarray)):
        data = np.asarray(data)
        im = Image.fromarray(data.astype('uint8'))

    resized_im = im.resize(flat, Image.NEAREST)
    resized_im = np.array(resized_im)
    try:
        resized_im = cv2.cvtColor(resized_im, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        pass

    liste = []

    for elem in resized_im:
        elem = elem.tolist()
        for e in range(len(elem)):
            if elem[e] == 0:
                elem[e] = 0.1
        liste.extend(elem)

    return liste
