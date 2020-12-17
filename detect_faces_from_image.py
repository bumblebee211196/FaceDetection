"""
detect_face.py: Description of what detect_face.py does.
"""

__author__ = "S Sathish Babu"
__date__ = "17/12/20 Thursday 11:20 AM"
__email__ = "sathish.babu@zohocorp.com"

import argparse

import cv2
import numpy as np


def detect_face(image):
    """Detect faces from the given IMAGE

    :param image: Image
    """
    net = cv2.dnn.readNetFromCaffe('resources/deploy.prototxt.txt',
                                   'resources/res10_300x300_ssd_iter_140000.caffemodel')
    (heigth, width) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image=cv2.resize(image, (300, 300)), size=(300, 300), scalefactor=1.0,
                                 mean=(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, heigth, width, heigth])
            (x1, y1, x2, y2) = box.astype('int')
            text = f'{confidence * 100 :.2f}%'
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            if y1 - 10 > 10:
                x, y = x1 + 5, y1 - 10
            else:
                x, y = x1 + 5, y1 + 10
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 255), 1)
    cv2.imshow('Ouput', image)
    cv2.waitKey(0)


PARSER = argparse.ArgumentParser('FaceDetection using Python and OpenCV')
PARSER.add_argument('-i', '--image', help='Path to the Image file', required=True, type=str)
ARGS = PARSER.parse_args()

detect_face(cv2.imread(ARGS.image))
