from enum import Enum

class Masks(Enum):
    YOLOV3 = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    YOLOV4 = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    YOLOV3TINY = [[3, 4, 5], [0, 1, 2]]
    # Anchors for YOLOV4TINY are deliberately wrong, in order to match the wrong masks in CFG for pretrained YoloV4-tiny
    # from AlexeyAB repository, and keep the functionality of the previous 2 repositories from TNTWEN and mystic123.
    # In this repository, we provide a flag to override this anchors!
    YOLOV4TINY = [[3, 4, 5], [0, 1, 2]]