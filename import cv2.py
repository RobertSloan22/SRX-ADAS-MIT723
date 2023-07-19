import cv2
import numpy as np
from gui_buttons import Buttons

# set the distance between headlights in feet
headlight_distance = 3

# set the distance threshold in feet for when a car is considered too close
distance_threshold = 15

# load the object detection model and set the minimum confidence threshold
model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')
min_confidence = 0.5

button = Buttons()
button.add_button("person", 0, 0)
button.add_button("truck", 0, 35)
button.add_button("car", 0, 70)
button.add_button("traffic light", 0, 105)
button.add_button("bicycle", 0, 140)
button.add_button("stop sign", 0, 175)
button.add_button("train", 0, 210)
button.add_button("bus", 0, 245)
button.add_button("motorbike", 0, 280)
button.add_button("aeroplane", 0, 315)
button.add_button("fire hydrant", 0, 350)
button.add_button("dog", 0, 385)
button.add_button("backpack", 0, 420)
colors = button.colors

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Objects list")
print(classes)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)

def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Frame", click_button)
while True:
    ret, frame = cap.read()

    active_buttons = button.active_buttons_list()

    # detect objects in the frame using the object detection model
    model.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
    output = model.forward()

    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > min_confidence:
            class_id = int(detection[1])
            if classes[class_id] in active_buttons:
                # get the bounding box coordinates and calculate the size of the bounding box
                box = detection[3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                bbox_width = endX - startX
                bbox_height = endY - startY

                # calculate the distance to the car assuming headlight_distance is 3 feet
                distance = headlight_distance / bbox_width

                # check if the car is too close
                if distance < distance_threshold:
                    alert_message = "{} is too close!".format(classes[class_id])
                    cv2.putText(frame, alert_message, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Calculate the distance between the object and the camera
        center = ((startX + endX) // 2, (startY + endY) // 2)
        distance_from_camera = distance(center, (frame.shape[1] // 2, frame.shape[0] // 2))
        distance_in_feet = distance_from_camera * 0.00328084  # 1 meter = 3.28084 feet

        # If the object is within 10 feet, display an alert message on the screen
        if distance_in_feet < 10:
            alert_message = "{} is {} feet away".format(classes[class_id], round(distance_in_feet, 2))
            cv2.putText(frame, alert_message, (startX, startY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # display the bounding box and label on the frame
        color = colors[class_id]
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, classes[class_id], (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    button.display_buttons(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

