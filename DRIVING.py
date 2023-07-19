import cv2
from gui_buttons import Buttons
import numpy as np
import time




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

car_cascade = cv2.CascadeClassifier('cars1.xml')

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)


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


cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

cv2.setMouseCallback("Frame", click_button)

while True:

    ret, frame = cap.read()

    active_buttons = button.active_buttons_list()

    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]


        color = colors[class_id]

        if class_name in active_buttons:
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, 255, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

         # Detects cars of different sizes in the input image
         # cars = car_cascade.detectMultiScale(gray, 1.1, 1)

         # To draw a rectangle in each cars
         # for (x, y, w, h) in cars:
   #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


   #button.display_buttons(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0)
    if key == 27:
        break


