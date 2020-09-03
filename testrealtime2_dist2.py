import cv2
import numpy as np
import time

# Load Yolo
net = cv2.dnn.readNet("weights/yolov3-tiny_best_62.weights", "cfg/yolov3-tiny.cfg",)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

FOV = 1150  # This variable should be changed if the estimated distance does not seem right.

# Increase the value of FOV if the estimated distance is too high and you want to reduce it
# Decurease the value of FOV if the estimated distance is too low and you want to increase it

# Loading image
cap = cv2.VideoCapture("IMG_5596.mp4")

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    # frame = cv2.flip(frame, 0)
    frame_id += 1

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    H, W, _ = frame.shape  # This calculates the width and height of the image
    # This puts lines on the screen that help visualize what is directly in front of the
    # car and the cars inside the lines will have their distances calculated.
    # cv2.line(frame, (int(W / 3), 0), (int(W / 3), H), (255, 0, 0), 2)
    # cv2.line(frame, (int(W / 1.5), 0), (int(W / 1.5), H), (255, 0, 0), 2)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                label + " " + str(round(confidence, 2)),
                (x, y - 30),
                font,
                2,
                color,
                3,
            )
            if (
                label == "car"
            ):  # If the detected object is a car then we calculate the distance
                mid = x + w / 2  # This is the mid point of where the car is
                if (
                    mid > W / 3 and mid < W / 1.5
                ):  # If the car is between the lines then calculate the distance. If the car is directly in front of the camera only then calculate the distance.
                    d = w / 2  # This is half the width of the car
                    degreePerPixel = (
                        FOV / W
                    )  # Over here it is calculated how many degree are represented a single pixel
                    ang = (
                        d * degreePerPixel
                    )  # This calculates the angle that is made by the car with the camera
                    ang = np.deg2rad(ang)  # We change the angle from degrees to radians

                    dist = d / np.tan(
                        ang
                    )  # This calculates the distance from the camera to the car in the front
                    dist = abs(dist)
                    # We put the distance inside the bounding box
                    cv2.putText(
                        frame,
                        "Dist: {:.1f}".format(dist),
                        (x, y + int(h - 5)),
                        font,
                        2,
                        (255, 0, 0),
                        3,
                    )

                    if (
                        dist < 10
                    ):  # If the distance is less than 5 meters then put text that says stop
                        cv2.putText(
                            frame, "STOP", (x, y + int(h / 2)), font, 2, (0, 0, 255), 3,
                        )
                    print(
                        "mid: {:.1f}".format(d),
                        "Ang: {:.1f}".format(ang),
                        "degreePerPixel: {:.1f}".format(degreePerPixel),
                        "Dist: {:.1f}".format(dist),
                    )

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
