import cv2
import numpy as np
from ultralytics import YOLO

# select model
# n , s , m , l , x : nano, small, medium, large, and xtra large models
# " " = object detection
# "cls" = classification
# "seg" = segmentation
# "pose" = human pose estimation 
model = YOLO("yolov8m-seg.pt")

# filepath to video or image
# alternatively, "0" for your webcam
videoSource = "0"  

# make predictions on source
# show: show the video when running
# stream: prevents text from accumulating in console
# save: saves video to your comnputer
# conf: minimum confidence to appear as detected
# classes: show what classes you would like to detect
results = model.predict(source=videoSource, show=True, stream=True, save=True, conf=0.30, classes = [0])

# run results as a loop of image frames
for result in results:
    frame = result.orig_img.copy()  # original result frame

    # check for segmentation masks
    if result.masks is not None and result.boxes is not None:
        masks = result.masks.data.cpu().numpy()  # grab segmentation masks
        num_masks = masks.shape[0] # find number of masks present
        class_ids = result.boxes.cls.cpu().numpy()  # find class ids for the bboxes
        class_names = result.names  # get class names

        # create mask for areas not covered by segmenation mask 
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # make the background grey
        background_color = np.array([128, 128, 128], dtype=np.uint8)  # Medium gray in BGR format

        # copy the results and make everything grey
        frame[:] = background_color

        # make each segmentation mask into a single combined mask 
        for i in range(num_masks):
            mask = masks[i].astype(np.uint8)
            
            class_id = int(class_ids[i])  # get the class id for a specific mask

            # check if object is a person, you could also say "if class_id == 0"
            if class_names[class_id] == 'person':
                # make humans red (BGR)
                color = np.array([0, 0, 0], dtype=np.uint8)  
                
                # fill in mask with your color
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[:, :, 0] = mask * color[0]  # Blue
                colored_mask[:, :, 1] = mask * color[1]  # Green
                colored_mask[:, :, 2] = mask * color[2]  # Red

                # Apply the red color only to the masked areas for humans
                frame[mask == 1] = colored_mask[mask == 1]

            # add to the total mask (binary, so take the maximum to "draw" on it)
            full_mask = np.maximum(full_mask, mask)

        
    else:
        # if there are no masks, leave the background and everything else grey
        frame[:] = np.array([128, 128, 128], dtype=np.uint8)  

    # show other window with edited display
    cv2.imshow("WHOA", frame)


    # rapidly press q to leave cleanly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# you know whar this does
cv2.destroyAllWindows()
