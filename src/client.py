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

Service_name = '/yolov3_detection_service'

class yolov3_detection_client:
    def __init__(self):
        rospy.wait_for_service(Service_name)
        self._image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback, queue_size=1)
        # self._image_pub = rospy.Publisher("/detection/results",Image,queue_size=10)
        self._bridge = CvBridge()
        self._yolov3_detection_client = rospy.ServiceProxy(Service_name, DetectionTask)

    def callback(self, data):
        print("yolov3_detection_client callback {}".format(rospy.Time.now()))
        req=DetectionTaskRequest(data)
        try:
            response = self._yolov3_detection_client(req)
            print(response.results)
            return response.results
        except rospy.ServiceException, e:
            print ("Service call failed: %s" %e)
            cv2.waitKey(1000)

def main(args):
    rospy.init_node('yolov3_detection_client', anonymous=True)
    client = yolov3_detection_client()
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)