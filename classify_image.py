#!/usr/bin/env python3

import cv2
import sys
import copy
import numpy as np
import cozmo
from imgclassification import ImageClassifier
from cozmo.util import degrees, Angle, Pose, distance_mm, speed_mmps
from skimage import io, feature, filters, exposure, color

try:
	from PIL import Image, ImageDraw, ImageFont
except ImportError:
	sys.exit('install Pillow to run this code')


def classify_image(image):
    img_clf = ImageClassifier()

    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    
    train_data = img_clf.extract_image_features(train_raw)
    img_clf.train_classifier(train_data, train_labels)
    
    predicted_label = img_clf.predict_labels(image)
    print(predicted_label)
    
    return(predicted_label)
	  
async def run(robot: cozmo.robot.Robot):
    '''The run method runs once the Cozmo SDK is connected.'''
    try:
        while True:
            event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)
            event.image.save("image.bmp")

            image = io.imread("image.bmp").astype(np.uint8)
            fd = filters.gaussian(color.rgb2gray(image), sigma=3)
            predicted_label = classify_image(fd)
            
            if predict_label == "plane":
                await robot.drive_straight(distance_mm(20), speed_mmps(20)).wait_for_completed()
                await robot.say_text("plane").wait_for_completed()
            elif predict_label == "truck":
                await robot.turn_in_place(degrees(90), speed=degrees(45)).wait_for_completed()
                await robot.say_text("truck").wait_for_completed()
            elif predict_label == "drone":
                await robot.turn_in_place(degrees(90), speed=degrees(45)).wait_for_completed()
                await robot.say_text("drone").wait_for_completed()
            
    except KeyboardInterrupt:
        print("")
        print("Exit requested by user")
    except cozmo.RobotBusy as e:
        print(e)



if __name__ == '__main__':
    cozmo.run_program(run, use_viewer = True, force_viewer_on_top = True)

