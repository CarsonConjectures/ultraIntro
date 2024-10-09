# ultraIntro


During graphics group, a series of changes to the code were made to demonstrate how to change the commands to do other tasks. 

The graphicsGroupPOSE.py code was not showing the original detections, and only the "blank canvas" screen, so I called the variable in the results loop to show the original frame:
```{txt}
# run results as a loop of image frames
for result in results:
    frame = result.orig_img.copy()  # original image frame

    frame
    # show live prediction (redundant, but allows for editing like in the SEGMENT file)
    cv2.imshow("classify blank canvas", frame)

    # rapidly press q to leave cleanly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

The classify code was not modified, but you can still change the model call to use the pretrained weights instead of the weights I had made:
```{txt}
model = YOLO("C:\\Users\\trogi\\Downloads\\last (1).pt")
```

The detect code was originally set to only detect humans, but removing the class restriction allows all classes that the model is trained to detect appear:
```{txt}
results = model.predict(source=videoSource, show=True, stream=True, save=True, conf = 0.30) #, classes = [0])
```

By far, the most modified script was the open vocab script, which allows users to change the detected class by describing the class verbally. Here are some examples we tested:

```{txt}
# just detect smartphones
model.set_classes(["SMARTPHONE"])

# detect smarphones and coffee cups
model.set_classes(["SMARTPHONE", "COFFEE CUP"])

# detect just TVs
model.set_classes(["TV"])

# detect just whiteboards
model.set_classes(["WHITEBOARD"])

# detect whiteboards and TVs, shown to be more accurate than one or the other
model.set_classes(["WHITEBOARD", "TV"])

# just detect mice, interpreted as computer mice
model.set_classes(["MOUSE"])

# detect computer and animal mice
model.set_classes(["COMPUTER MOUSE", "MOUSE MOUSE"])

# just detect mirrors, or at least try
model.set_classes(["mirror"])
```
