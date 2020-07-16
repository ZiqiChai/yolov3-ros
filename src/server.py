#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys

import cv2

import roslib
roslib.load_manifest('yolov3-ros')

import rospy
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from universal_msgs.srv import DetectionTask, DetectionTaskResponse, DetectionTaskRequest

class yolov3_detection_server:
    def __init__(self):
        self._yolov3_detection_server = rospy.Service('/yolov3_detection_service', DetectionTask, self.callback)
        self._bridge = CvBridge()

    def callback(self, req):
        print("yolov3_detection_server callback {}".format(rospy.Time.now()))
        try:
            cv_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            print(e)
        res = DetectionTaskResponse()
        res.results.data = [0,0.1,0.2,0.3,0.4,1,0.5,0.6,0.7,0.8]
        return res

def main(args):
    rospy.init_node('yolov3_detection_server', anonymous=True)
    server = yolov3_detection_server()
    try:
        rospy.spin()
    except KeyboardInterrupt:
       print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)