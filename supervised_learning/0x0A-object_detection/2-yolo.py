#!/usr/bin/env python3
"""
Class Yolo that uses the Yolo v3 algorithm to perform object detection
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


class Yolo:
    """
    Class constructor
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Defining class, that performs object detection
        """
        self.model = load_model(model_path)
        self.class_names = open(classes_path).read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Defining outputs
        Returns a tuple of (boxes, box_confidences, box_class_probs)
        """
        box_confidences = []
        boxes = []
        box_class_probs = []

        for i in range(len(outputs)):
            box_confidences.append(tf.math.sigmoid(outputs[i][:, :, :, 4:5]))

        for i in range(len(outputs)):
            boxes.append(outputs[i][:, :, :, :4])
        for i in range(len(outputs)):
            box_class_probs.append(tf.math.sigmoid(outputs[i][:, :, :, 5:]))
        with tf.Session() as sess:
            box_confidences = sess.run(box_confidences)
            box_class_probs = sess.run(box_class_probs)
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Contains the processed boundary boxes for each output, respectively
        Returns a tuple of (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = np.concatenate([box.reshape(-1, 4) for box in boxes])
        reshaped_boxes = [box.reshape(-1, 80) for box in box_class_probs]
        cat_box_classes = np.concatenate([box for box in reshaped_boxes])
        box_classes = cat_box_classes.argmax(axis=1)
        box_scores = np.concatenate([box_conf.reshape(-1) for box_conf in box_confidences])
        box_cl_max = cat_box_classes.max(axis=1)
        box_scores *= box_cl_max
        indecies = np.where(box_scores > self.class_t)
        return (filtered_boxes[indecies], box_classes[indecies], box_scores[indecies])
