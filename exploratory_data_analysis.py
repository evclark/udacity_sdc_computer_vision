'''
File containing useful classes and functions for exploratory data analysis
'''

import cv2
import math
import numpy as np
import random
import tensorflow as tf
import os

from dataclasses import InitVar, dataclass, field
from collections import Counter
from typing import Union, List

#### Constants ####
CYAN = (255, 0, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
CLASS_TO_LABEL_MAP = {1: "vehicle", 2: "pedestrian", 4: "cyclist"}
LABEL_TO_CLASS_MAP = {value: key for key, value in CLASS_TO_LABEL_MAP.items()}
DEFAULT_CLASS_TO_COLOR_MAP = {1: CYAN, 2: GREEN, 4: RED}

#### Utility Functions ####
def calculateSharpness(img: np.array) -> float:
    '''
    Calculates images sharpness based on the FIXME metric
    '''
    return cv2.Laplacian(img, cv2.CV_64F).var()

#### Utility Classes ####
@dataclass
class GroundTruthAnnotator:
    '''
    Class for annotating ground truth bounding boxes around classes visible in
    the waymo open dataset imagery
    '''
    class_color_map: dict = field(default_factory=lambda: 
                                                     DEFAULT_CLASS_TO_COLOR_MAP)
    annotation_box_thickness: int = 1
    mosiac_cols: int = 5
    mosiac_width_px: int = 1000

    def _convert_to_pixel_coords(self,
                                 x1: float, y1: float, 
                                 x2: float, y2: float,
                                 img_shape: np.array) -> np.array:
        '''
        Converts bounding box coordinates from image relative [0.0-1.0] to pixel
        coordinates 
        '''
        return (int(x1 * img_shape[1]),
                int(y1 * img_shape[0]),
                int(x2 * img_shape[1]),
                int(y2 * img_shape[0]))

    def get_grid_offset(self, img_idx: int, img_height: int, img_width: int):
        '''
        Calculate the pixel offset for each sub image to go into the mosiac 
        created by display_instances()
        '''
        grid_row = img_idx // self.mosiac_cols
        grid_col = img_idx % self.mosiac_cols
        return grid_row * img_height, grid_col * img_width

    def annotate_ground_truth(self, 
                              img: np.array, 
                              bounding_boxes: np.array, 
                              classes: np.array
                              ) -> np.array: 
        '''
        Draw bounding box annotation around ground truth instances in an image
        '''
        #Annotate a copy of the image, not the original
        annotated_img = img.copy()

        #Draw in the bounding boxes
        for i, box in enumerate(bounding_boxes):
            #Get the color for the class, or if one doesn't exist yet create one
            truth_class = classes[i]
            if truth_class not in self.class_color_map:
                self.class_color_map[truth_class] = [random.randint(0, 255),
                                                     random.randint(0, 255),
                                                     random.randint(0, 255)]
            color = self.class_color_map[classes[i]]
            
            #Draw the annotated bounding box
            y1, x1, y2, x2 = self._convert_to_pixel_coords(*box, img.shape)
            cv2.rectangle(annotated_img, 
                          (x1, y1), 
                          (x2, y2), 
                          color, 
                          self.annotation_box_thickness)
        
        return annotated_img

    def batch_annotate_ground_truth(self, 
                                    batch: tf.data.Dataset, #Batch of images with ground truth labels
                                    filename_suffix: str = "" #Suffix to add to mosiac file name 
                                    ) -> str: 
        """
        Takes a batch of images, combines them into a mosiac and annotates them 
        with ground truth labels, saves it, and returns the file path
        """    
        #Convert batch to list to make it easier to work with
        batch = list(batch.as_numpy_iterator())

        #Initialize the blank mosiac we will fill in with annotated images
        batch_size = len(batch)
        mosaic_rows = math.ceil(batch_size / self.mosiac_cols)
        first_img = batch[0]["image"]
        img_height, img_width, img_depth = first_img.shape
        mosaic = np.zeros((mosaic_rows * img_height, 
                           self.mosiac_cols * img_width, 
                           img_depth),
                           first_img.dtype)    

        #Fill the mosiac with annotated ground truth images
        for i, instance in enumerate(batch):
            #Parse out the image, ground truth bounding boxes and classes
            img = instance["image"] 
            bounding_boxes = instance["groundtruth_boxes"]
            classes = instance["groundtruth_classes"]
            
            #Get an annotated version of the image
            annotated_img = self.annotate_ground_truth(
                                                   img, bounding_boxes, classes)

            #Add the annotated image to the mosiac
            offset_y, offset_x = self.get_grid_offset(i, img_height, img_width)
            mosaic[offset_y : offset_y + img_height, 
                   offset_x : offset_x + img_width,
                   :] = annotated_img 

        #Create the eda_output folder if it doesn't already exist
        if not os.path.exists("eda_output"):
            os.mkdir("eda_output")
            
        #Save the mosaic and return the file path
        mosiac_file_path = ("eda_output/groundtruth_mosaic_%s.png" 
                                                              % filename_suffix)
        cv2.imwrite(mosiac_file_path, mosaic)
        return mosiac_file_path
        
@dataclass 
class MinMedianMaxImgExample:
    '''
    Helper class for holding images that are statistically meaningful to  
    a group of images, e.g. the min, median, and max in terms of some image 
    attribute like brightness, sharpness, or contrast
    '''
    attribute_name: str
    min_value: float = np.inf
    max_value: float = -np.inf
    median_value: float = np.inf
    min_img = None
    median_img = None
    max_img = None

    def updateMinAndMax(self, value: float, img: np.array) -> None:
        if value < self.min_value:
            self.min_value = value
            self.min_img = img

        if value > self.max_value:
            self.max_value = value
            self.max_img = img

    def updateMedian(self, value: float, img: np.array) -> None:
        self.median_value = value
        self.median_img = img

    def saveExampleImages(self, attribute_name: str) -> List[str]:
        img_path_map = {
            "min_%s.png" % attribute_name: self.min_img,
            "median_%s.png" % attribute_name: self.median_img,
            "max_%s.png" % attribute_name: self.max_img
        }

        for path, img in img_path_map.items():
            cv2.imwrite(path, img)
        

@dataclass 
class StatisticsAggregator:
    '''
    Class for calculating/aggregating various statistics about images in the
    waymo open dataset
    '''

    def __init__(self) -> None:
        #Initialize all member variables as empty until the relevant
        #calculate*() method is called
        self.class_freqs = {key: [] for key in CLASS_TO_LABEL_MAP.keys()}
        self.class_widths = {key: [] for key in CLASS_TO_LABEL_MAP.keys()}
        self.class_heights = {key: [] for key in CLASS_TO_LABEL_MAP.keys()}
        self.class_bounding_box_centers = {key: [] for key in 
                                                      CLASS_TO_LABEL_MAP.keys()}
        self.img_attributes = ["mean_brightness", "contrast", "sharpness"]
        self.img_statistics = {key: [] for key in self.img_attributes}
        self.img_examples = {key: MinMedianMaxImgExample(key) for key in 
                                                            self.img_attributes}

    def calculateAllStats(self, batch: tf.data.Dataset) -> None:
        self.calculateFreqStats(batch)
        self.calculateSizeAndPosStats(batch)
        self.calculateImageStats(batch)

    def calculateFreqStats(self, batch: tf.data.Dataset) -> None:
        '''
        Computes class frequency statistics of objects in the batch and adds 
        them to the aggregate statistics
        '''
        for elem in batch:
            class_counts = Counter(elem["groundtruth_classes"])
            for class_idx in self.class_freqs.keys():
                self.class_freqs[class_idx].append(class_counts[class_idx])

    def calculateSizeAndPosStats(self, batch: tf.data.Dataset) -> None:
        '''
        Computes size and position statistics of objects in the batch and adds 
        them to the aggregate statistics
        '''
        for elem in batch:
            bounding_boxes = elem["groundtruth_boxes"]
            for box_idx, box in enumerate(bounding_boxes):
                y1, x1, y2, x2 = box
                width = x2 - x1
                height = y2 - y1 
                center = (np.mean([x1, x2]), np.mean([y1, y2]))
                class_idx = elem["groundtruth_classes"][box_idx]
                self.class_widths[class_idx].append(width)
                self.class_heights[class_idx].append(height)
                self.class_bounding_box_centers[class_idx].append(center)  

    def calculateImageStats(self, batch) -> None:
        '''
        Calculates statistics about images in the batch, e.g. brightness, 
        sharpness, contrast and adds them to the aggregate statistics
        '''
        for elem in batch:
            img = elem["image"]
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            brightness = np.mean(gray_img.flatten())
            contrast = gray_img.std()
            sharpness = calculateSharpness(gray_img)
            
            self.img_statistics["mean_brightness"].append(brightness)
            self.img_statistics["contrast"].append(contrast)
            self.img_statistics["sharpness"].append(sharpness)

            self.img_examples["mean_brightness"].updateMinAndMax(brightness, img)
            self.img_examples["contrast"].updateMinAndMax(contrast, img)
            self.img_examples["sharpness"].updateMinAndMax(sharpness, img)

            if np.median(self.img_statistics["mean_brightness"]) == brightness:
                self.img_examples["mean_brightness"].updateMedian(brightness, img)
            if np.median(self.img_statistics["contrast"]) == contrast:
                self.img_examples["contrast"].updateMedian(contrast, img)
            if np.median(self.img_statistics["sharpness"]) == sharpness:
                self.img_examples["sharpness"].updateMedian(sharpness, img)

