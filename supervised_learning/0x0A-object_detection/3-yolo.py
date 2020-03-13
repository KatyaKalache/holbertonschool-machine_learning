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
        box_scores = np.concatenate([box_conf.reshape(-1) for
                                     box_conf in box_confidences])
        box_cl_max = cat_box_classes.max(axis=1)
        box_scores *= box_cl_max
        indecies = np.where(box_scores > self.class_t)
        return (filtered_boxes[indecies],
                box_classes[indecies], box_scores[indecies])

    def iou(self, box1, box2):
        """
        Calculates intersection over union
        """
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        area_intersect = (xB - xA + 1) * (yB - yA + 1)
        area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        iou = area_intersect / float(area_box1 + area_box2 - area_intersect)
        return iou

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Predicts bounding boxes ordered by class and box score
        Returns a tuple of
        (box_predictions, predicted_box_classes, predicted_box_scores)
        """
        inds_sorted = np.argsort(box_classes)
        box_scores_sorted = box_scores[inds_sorted]
        box_classes_sorted = box_classes[inds_sorted]
        filtered_boxes_sorted = filtered_boxes[inds_sorted]
        idx = []
        for i in range(len(box_scores_sorted)):
            for j in range(i + 1, len(box_scores_sorted)):
                if (box_classes_sorted[j] == box_classes_sorted[i]):
                    union_score = self.iou(filtered_boxes_sorted[j],
                                           filtered_boxes_sorted[i])
                    if union_score < self.nms_t:
                        idx.append(j)
        box_predictions = filtered_boxes_sorted[idx]
        predicted_box_classes = box_classes_sorted[idx]
        predicted_box_scores = box_scores_sorted[idx]
        return(box_predictions, predicted_box_classes, predicted_box_scores)
