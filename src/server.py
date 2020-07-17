#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import os
import sys
import platform
import time
import datetime
import argparse

PyTorch_YOLOv3_Py27_path = os.path.abspath(os.path.dirname(__file__)) + "/../PyTorch_YOLOv3_Py27"
sys.path.append(PyTorch_YOLOv3_Py27_path)

from models import *
from utils.utils import *
from utils.datasets import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
# avoid "RuntimeError: main thread is not in main loop"
plt.switch_backend('agg')

# avoid
from PIL import Image as PILImage
import cv2

import roslib
roslib.load_manifest('yolov3-ros')

import rospy
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from universal_msgs.srv import DetectionTask, DetectionTaskResponse, DetectionTaskRequest

def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # import PIL.Image as PILImage
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = PILImage.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


class yolov3_detection_server:
    def __init__(self, parser):
        self._parser = argparse.ArgumentParser()
        self._parser = parser
        self._opt = self._parser.parse_args()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up model
        self.model = Darknet(self._opt.model_def, img_size=self._opt.img_size).to(self.device)

        if self._opt.weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self._opt.weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self._opt.weights_path))

        self.model.eval()  # Set in evaluation mode
        self.classes = load_classes(self._opt.class_path)  # Extracts class labels from file
        self._yolov3_detection_server = rospy.Service('/yolov3_detection_service', DetectionTask, self.callback)
        self._image_pub = rospy.Publisher("/detection/results",Image,queue_size=10)
        self._bridge = CvBridge()

    def callback(self, req):
        excute_time = rospy.Time.now()
        print("\nyolov3_detection_server callback {}".format(excute_time))
        cv_image = np.zeros((req.image.height, req.image.width, 3), dtype=np.int8)
        try:
            cv_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            print(e)

        single_input_img = transforms.ToTensor()(PILImage.fromarray(cv2.cvtColor(cv_image.copy(), cv2.COLOR_BGR2RGB)))
        single_input_img, _ = pad_to_square(single_input_img, 0)
        single_input_img = resize(single_input_img, self._opt.img_size)

        print("Performing object detection:")
        prev_time = time.time()
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        single_input_img = Variable(single_input_img.type(Tensor))
        single_input_img = torch.unsqueeze(single_input_img, dim=0)

        # Get detections
        with torch.no_grad():
            detections = self.model(single_input_img)
            # print(type(detections))
            # print(detections)
            detections = non_max_suppression(detections, self._opt.conf_thres, self._opt.nms_thres)
            # print(type(detections))
            # print(detections)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("Inference Time: %s" % (inference_time))

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        # Create plot
        img = cv2.cvtColor(cv_image.copy(), cv2.COLOR_BGR2RGB)
        figure = plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections

        # print(type(detections))
        # print(detections)

        # when the source has only one input image, we should use index 0 to convert data type of detections
        # Convert the following:
        # [tensor([   [ 17.0720, 143.8836, 103.6933, 200.3457,   1.0000,   1.0000,   0.0000],
        #             [297.4258, 142.2059, 389.5064, 205.5459,   1.0000,   1.0000,   0.0000],
        #             [184.6890, 181.4856, 258.8690, 232.0957,   1.0000,   1.0000,   0.0000]
        #             ])
        # ]
        # To:
        # tensor([    [ 17.0720, 143.8836, 103.6933, 200.3457,   1.0000,   1.0000,   0.0000],
        #             [297.4258, 142.2059, 389.5064, 205.5459,   1.0000,   1.0000,   0.0000],
        #             [184.6890, 181.4856, 258.8690, 232.0957,   1.0000,   1.0000,   0.0000]
        #             ])
        # This is caused by removing the img_detections list and its extend function.
        # When not using index of img_detections, the data type just mismatched.

        detections = detections[0]
        detection_list=list()
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, self._opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t + Label: %s, Conf: %.5f" % ((self.classes[int(cls_pred)]).replace('\n', '').replace('\r', ''), cls_conf.item()))
                box_w = x2 - x1
                box_h = y2 - y1
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=(self.classes[int(cls_pred)]).replace('\n', '').replace('\r', ''),
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )
                detection_list.extend([int(cls_pred), x1, y1, box_w, box_h])

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())

        # figure_name_to_save = self._opt.outputs_folder + "/" + str(excute_time) +".png"
        figure_name_to_save = self._opt.outputs_folder + "/results.png"
        plt.savefig(figure_name_to_save, bbox_inches="tight", pad_inches=0.0)
        plt.close()

        image = fig2data(figure)
        try:
            self._image_pub.publish(self._bridge.cv2_to_imgmsg(image, "rgba8"))
        except CvBridgeError as e:
            print(e)



        res = DetectionTaskResponse()
        res.results.data = detection_list
        # _image_pub.publish()
        # rospy.sleep(0.050)
        return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="../PyTorch_YOLOv3_Py27/assets", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="../PyTorch_YOLOv3_Py27/config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="../PyTorch_YOLOv3_Py27/checkpoints/yolov3_ckpt_399.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="../PyTorch_YOLOv3_Py27/data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.2, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--outputs_folder", type=str, default="outputs", help="path to save detect results")
    opt = parser.parse_args()
    print_parser(opt)

    if not os.path.exists(opt.outputs_folder):
        os.makedirs(opt.outputs_folder)


    rospy.init_node('yolov3_detection_server', anonymous=True)
    server = yolov3_detection_server(parser)
    try:
        rospy.spin()
    except KeyboardInterrupt:
       print("Shutting down")
    cv2.destroyAllWindows()