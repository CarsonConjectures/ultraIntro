import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics import YOLOWorld



# filepath to video or image
# alternatively, "0" for your webcam
videoSource = "0"  # Or path to a video file

# choose open vocab model
model = YOLOWorld("yolov8m-worldv2.pt")
#describe the objects you want to detect
model.set_classes(["SMARTPHONE"])

# make predictions on source
# show: show the video when running
# stream: prevents text from accumulating in console
# save: saves video to your comnputer
results = model.predict(source=videoSource, show=True, stream=True, save=True)

# run results as a loop of image frames
for result in results:
    frame = result.orig_img.copy()  # original image frame

    # show live prediction (redundant, but allows for editing like in the SEGMENT file)
    cv2.imshow("classify blank canvas", frame)

    # rapidly press q to leave cleanly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
