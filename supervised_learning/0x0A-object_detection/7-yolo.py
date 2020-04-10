#!/usr/bin/env python3
"""
Class Yolo that uses the Yolo v3 algorithm to perform object detection
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2


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

    @staticmethod
    def load_images(folder_path):
        """
        Loads images
        Returns: a tuple of (images, image_paths)
        """
        images = []
        file_paths = []
        for fn in os.listdir(folder_path):
            path = folder_path + '/' + fn
            images.append(cv2.imread(path))
            file_paths.append(path)
        return(images, file_paths)

    def preprocess_images(self, images):
        """
        Resize the images with inter-cubic interpolation
        Rescale all images to have pixel values in the range [0, 1]
        Returns a tuple of (pimages, image_shapes)
        """
        image_shapes = np.empty((len(images), 2))
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]
        pimages = np.empty((len(images), input_h, input_w, 3))

        for i, im in enumerate(images):
            image_shapes[i][0] = im.shape[0]
            image_shapes[i][1] = im.shape[1]
            pimages[i] = cv2.resize(im / 255,
                                    (input_h, input_w),
                                    interpolation=cv2.INTER_CUBIC)
        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes,
        class names, and box scores
        """
        for i, box in enumerate(boxes):
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            label = self.class_names[box_classes[i]] + '{:.2f}'.format(
                box_scores[i])
            rect = cv2.rectangle(image, (x, y), (w, h),
                                 255, 2, cv2.LINE_AA)
            image = cv2.putText(rect, label, (x-5, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 1)
        cv2.imshow(file_name, image)
        if not os.path.isdir('./detections/'):
            os.mkdir('./detections/')
        if cv2.waitKey() == 115:
            cv2.imwrite('./detections/{}'.format(file_name), image)
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Predicts images from folder_path
        Returns: a tuple of (predictions, image_paths)
        """
        image_paths = []
        images, file_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)
        predictions = self.model.predict(pimages)
        num_im = len(pimages)
        for i in range(num_im):
            print(i)
        return(predictions, image_paths)
        
