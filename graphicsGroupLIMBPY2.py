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
model = YOLO("yolov8m-pose.pt")

# filepath to video or image
# alternatively, "0" for your webcam
videoSource = "0"  # Or path to a video file


# make predictions on source
# show: show the video when running
# stream: prevents text from accumulating in console
# save: saves video to your comnputer
# conf: minimum confidence to appear as detected
# classes: show what classes you would like to detect
results = model.predict(source=videoSource, show=False, stream=True, save=False, conf = 0.30, classes = [0])

# run results as a loop of image frames
for result in results:
    frame = result.orig_img.copy()  # original image frame

    # determine crosshair lines
    frame_height, frame_width, _ = frame.shape
    center_x = frame_width // 2
    center_y = frame_height // 2

    # assume no human until human appears
    targets = {"human": False, "human head": False}

    # check and evaluate detections
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy() 
        keypoints_array = result.keypoints.xy.cpu().numpy() 

        for i in range(len(boxes)):
            #bbox_x1,bbox_y1,bbox_x2,bbox_y2 = boxes[i]
            #bbox_center_x = int((bbox_x1 + bbox_x2) / 2)
            #bbox_center_y = int((bbox_y1 + bbox_y2) / 2)

            # threshold on how close an object needs to be to be 'in its sights'
            closeness_threshold = 100

            # euclid distance human from crossairs
            #human_distance = np.hypot(bbox_center_x - center_x, bbox_center_y - center_y)
            #if human_distance < closeness_threshold:
            #    targets["human"] = True
            person_keypoints = keypoints_array[i]

            for keypoint in person_keypoints:
                keypoint_x, keypoint_y = keypoint
                keypoint_distance = np.hypot(keypoint_x - center_x, keypoint_y - center_y)
                if keypoint_distance < closeness_threshold:
                    targets["human"] = True
                    break 

            person_keypoints = keypoints_array[i]
            head_x, head_y = person_keypoints[0]


            # euclid distance head from crossairs
            head_distance = np.hypot(head_x - center_x, head_y - center_y)
            if head_distance < closeness_threshold:
                targets["human head"] = True

    # make grayscale frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    #gray_frame = cv2.merge([gray_frame, gray_frame, gray_frame])
    gray_frame = cv2.merge([np.zeros_like(gray_frame), gray_frame, np.zeros_like(gray_frame)])  
    frame = gray_frame

    crosshair_color = (0,0,0) #(0,0,255)
    crosshair_thickness = 2

    # draw crosshair on frame
    cv2.line(frame, (center_x, 0+frame_height//2-frame_height//8), (center_x, frame_height//2+frame_height//8),crosshair_color,crosshair_thickness)
    cv2.line(frame, (0+frame_width//2- frame_width//8, center_y), (frame_width//2 + frame_width//8, center_y),crosshair_color,crosshair_thickness)


    screen_font = cv2.FONT_HERSHEY_PLAIN 

    # originally (0 0 255) and (0 255 0)
    if targets["human head"] or targets["human"]:
        cv2.putText(frame, "HUMAN IN CROSSHAIRS", (10, 30),
            screen_font, 1, (0, 0, 255), 2) 
    else:
        cv2.putText(frame, "HUMAN NOT IN CROSSHAIRS", (10, 30),
            screen_font, 1, (0, 0, 0), 2)
    if targets["human head"]:
        cv2.putText(frame, "HUMAN FACE IN CROSSHAIRS", (10, 60),
            screen_font, 1, (0, 0, 255), 2) 
    else:
        cv2.putText(frame, "HUMAN FACE NOT IN CROSSHAIRS", (10, 60),
            screen_font, 1, (0, 0, 0), 2)
        
    if targets["human head"]:
        cv2.putText(frame, "PRESCRIPTION: CAPSAICIN SPRAY", (10, 90),
            screen_font, 1, (0, 0, 0), 2)
    elif targets["human"]:
        cv2.putText(frame, "PRESCRIPTION: LOW POWER ELECTROLASER", (10, 90),
            screen_font, 1, (0, 0, 0), 2)
    else:
        cv2.putText(frame, "PRESCRIPTION: ANTI-MATERIAL LASER", (10, 90),
            screen_font, 1, (0, 0, 0), 2)


    # show live prediction (redundant, but allows for editing like in the SEGMENT file)
    cv2.imshow("classify blank canvas", frame)

    # rapidly press q to leave cleanly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
