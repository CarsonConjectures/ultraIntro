import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics import YOLOWorld


# select model
# n , s , m , l , x : nano, small, medium, large, and xtra large models
# " " = object detection
# "cls" = classification
# "seg" = segmentation
# "pose" = human pose estimation 
model = YOLO("yolov8m.pt")
#model = YOLO("SECONDSMALLWORK3FIX.pt")
#d:\move to play destiny\documents 6 24 2024\CSCE PROJ\A FILE FOR WEIGHTS\SECONDSMALLWORK3FIX.pt

# filepath to video or image
# alternatively, "0" for your webcam
videoSource = "0"  # Or path to a video file


# make predictions on source
# show: show the video when running
# stream: prevents text from accumulating in console
# save: saves video to your comnputer
# conf: minimum confidence to appear as detected
# classes: show what classes you would like to detect
results = model.predict(source=videoSource, show=True, stream=True, save=True, conf = 0.30, classes = [0])

# run results as a loop of image frames
for result in results:
    frame = result.orig_img.copy()  # original image frame

    # show live prediction (redundant, but allows for editing like in the SEGMENT file)
    cv2.imshow("classify blank canvas", frame)

    # rapidly press q to leave cleanly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
