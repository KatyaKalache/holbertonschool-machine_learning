#!/usr/bin/env python3
"""
Class Yolo that uses the Yolo v3 algorithm to perform object detection
"""
from tensorflow.keras.models import load_model


class Yolo:
    """
    Defining class, that performs object detection
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor
        """
        self.model = load_model(model_path) 
        self.class_names = open(classes_path).read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
        
