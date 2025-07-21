from ultralytics.utils.plotting import Annotator, colors
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import deprecation_warn

import torch
import numpy as np
import cv2

import sys
from copy import deepcopy
import json
import os
from pathlib import Path


class args:
    """
    This class contains configuration parameters for the vehicle counting system using the YOLO model and various tracking approaches.

    Attributes:
        source (str): Filename of the video to perform counting on.
                      Need to be set.
        name (str): Name of the folder for the current experiment results.
                    Need to be set.
        yolo_model (Path): Path to the YOLO model file.
                           Default is 'yolov8n.pt'.
        reid_model (Path): Path to the re-identification model file used if the tracker employs appearance description of objects.
                           Examples include 'osnet_x0_25_market1501.pt', 'mobilenetv2_x1_4_msmt17.engine', etc.
        tracking_method (str): Method used for tracking. Options include 'bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', and 'hybridsort'.
        imgsz (list): Input size of the frames.
                      Default is [640].
        conf (float): Confidence threshold for detection.
                      Default is 0.6.
        iou (float): Intersection over Union (IoU) threshold.
                     Default is 0.7.
        device (str): Device used for running the model (GPU by default).
                      Default is ''.
        show (bool): Whether to display the video scene. Not supported in Google Colab.
                     Default is False.
        save (bool): Whether to save the videos illustrating the tracking results.
                     Default is True.
        classes (list): List of class indices to detect.
                        Default is [1, 2, 3, 5, 7] (vehicles).
        project (str): Folder to save the tracking results.
                       Default is 'runs/count'.
        exist_ok (bool): Whether to overwrite existing results.
                         Default is True.
        half (bool): Whether to use half-precision (16-bit floating-point format) to reduce memory consumption.
                     Default is False.
        vid_stride (int): Frame stride, e.g., process all frames with stride=1 or process every other frame with stride=2.
                          Default is 1.
        show_labels (bool): Whether to display labels (e.g., car, truck, bus) in the saved video results.
                            Default is True.
        show_conf (bool): Whether to display confidence scores of detections.
                          Default is False.
        save_txt (bool): Whether to save results in a text file format.
                         Default is False.
        save_id_crops (bool): Whether to save tracking results for each object in frames.
                              Default is True.
        save_mot (bool): Whether to save tracking results in a report file.
                         Default is True.
        line_width (int): Line width of the bounding boxes.
                          Default is None.
        per_class (bool): Whether to count per class.
                          Default is True.
        verbose (bool): Whether to enable verbose logging.
                        Default is False.
        counting_approach (str): Approach for counting vehicles. Options include 'detection_only', 'tracking_without_line', 'tracking_with_line_vicinity', 'tracking_with_line_crossing', 'tracking_with_line_crossing_vicinity' ,  'tracking_with_two_lines'.
                                 Default is 'tracking_with_line_vicinity'.
        line_point11 (tuple): Coordinates of the first point of the first line. Values between 0 and 1 indicate percentages.
                              For example, (0.4, 0.0) means 40% of the frame width (pixel 0.4 * image width) and 0% of the frame height (pixel 0).
                              When masking the video frames with included_box, it becomes 0.4 * new width after mask.
        line_point12 (tuple): Coordinates of the second point of the first line. Values between 0 and 1 indicate percentages.
                              For example, (0.3, 1.0) means 30% of the frame width (pixel 0.3 * image width) and 100% of the frame height (pixel image height).
        line_vicinity (float): Vicinity of the line for counting. This argument is used in the 'detection_only' or 'tracking_with_line' counting approaches and ignored otherwise ('tracking_without_line' or 'tracking_with_two_lines').
                               Default is 0.1.
        line_point21 (tuple): Coordinates of the first point of the second line. Values between 0 and 1 indicate percentages.
                              For example, (0.6, 0.0) means 60% of the frame width (pixel 0.6 * image width) and 0% of the frame height (pixel 0).
                              This argument is considered only in the 'tracking_with_two_lines' counting approach and ignored otherwise.
        line_point22 (tuple): Coordinates of the second point of the second line. Values between 0 and 1 indicate percentages.
                              For example, (0.7, 1.0) means 70% of the frame width (pixel 0.7 * image width) and 100% of the frame height (pixel image height).
                              This argument is considered only in the 'tracking_with_two_lines' counting approach and ignored otherwise.
        use_mask (bool): Whether to use a mask for preprocessing. If set to False, 'visualize_masked_frames' and 'included_box' arguments will be ignored.
                         If set to True, the percentages for 'line_point11', 'line_point12', 'line_point21', and 'line_point22' will be transformed to pixel values with respect to the included_box.
                         Default is False.
        visualize_masked_frames (bool): Whether to visualize masked frames.
                                        Default is True.
        included_box (list): Box coordinates for masking, specified as percentages between -1 and 1. For example, [0.1, 0.2, -0.2, -0.1] indicates:
                             - The first two values (0.1, 0.2) represent the TOP-LEFT point of the included rectangle when using a mask for frames.
                               This point is 10% of the width and 20% of the height.
                             - The last two values (-0.2, -0.1) represent the BOTTOM-RIGHT point of the included rectangle after masking.
                               This point is 80% of the width and 90% of the height.
    """

    source = "kech.mp4"
    name = "kech"
    yolo_model = Path("yolov8n.pt")
    reid_model = Path("osnet_x0_25_msmt17.pt")
    tracking_method = "ocsort"
    imgsz = [640]
    conf = 0.6
    iou = 0.7
    device = ""
    show = False
    save = True
    classes = [1, 2, 3, 5, 7]
    project = "runs/count"
    exist_ok = True
    half = False
    vid_stride = 1
    show_labels = True
    show_conf = False
    save_txt = False
    save_id_crops = True
    save_mot = True
    line_width = None
    per_class = True
    verbose = False
    counting_approach = "tracking_with_line_vicinity"
    line_point11 = (0.4, 0.0)
    line_point12 = (0.3, 1.0)
    line_vicinity = 0.01
    line_point21 = (0.6, 0.0)
    line_point22 = (0.7, 1.0)
    use_mask = False
    visualize_masked_frames = True
    included_box = [0.1, 0.2, -0.2, -0.1]
    save_csv_count = False


class Annotator_for_counting(Annotator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def draw_line_with_two_points(
        self, line_point1, line_point2, color=(255, 255, 255), width=None
    ):
        """
        Draws a line given two points on the image.

        Args:
            line_point1 (tuple): Coordinates of the first point (x1, y1).
            line_point2 (tuple): Coordinates of the second point (x2, y2).
            color (tuple, optional): RGB color tuple for the line. Default is white (255, 255, 255).
            width (int, optional): Width of the line. Default is the line width specified during initialization.

        Note:
            This method modifies the image in place.
        """

        width = width or self.lw

        if self.pil:
            self.draw.line([line_point1, line_point2], fill=color, width=width)
        else:
            cv2.line(
                self.im,
                line_point1,
                line_point2,
                color,
                thickness=width,
                lineType=cv2.LINE_AA,
            )


class counter_YOLO(YOLO):
    pass
