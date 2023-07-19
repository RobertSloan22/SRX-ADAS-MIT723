import cv2
from gui_buttons import Buttons
import numpy as np
import math


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

# Load reference image and calculate focal length
ref_image = cv2.imread("BMW.png")
gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
car_cascade = cv2.CascadeClassifier("cars1.xml")
cars = car_cascade.detectMultiScale(gray_ref, 1.1, 5)
ref_distance = 195  # 15 feet = 180 inches
ref_width = 72  # 6 feet = 72 inches
focal_length = (ref_distance * ref_width) / cars[0][2]
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Load radar image
radar_image = cv2.imread("radarblue.png", cv2.IMREAD_UNCHANGED)
radar_image = cv2.resize(radar_image, (640, 480))

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Frame", click_button)
# Load radar image
radar_image = cv2.imread("radarblue.png", cv2.IMREAD_UNCHANGED)
radar_image = cv2.resize(radar_image, (640, 480))

while True:
    ret, frame = cap.read()
    
    # Draw radar field of view
    radar.draw_radar(frame)

    active_buttons = button.active_buttons_list()

    # Resize radar image to match size of frame
    resized_radar = cv2.resize(radar_image, (frame.shape[1], frame.shape[0]))

    # Blend radar image with frame
    alpha = 0.7
    beta = 1 - alpha
    blended = cv2.addWeighted(frame, alpha, resized_radar, beta, 0)

    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        color = colors[class_id]

        if class_name in active_buttons:
            cv2.putText(blended, class_name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, 255, 2)
            cv2.rectangle(blended, (x, y), (x + w, y + h), 255, 2)

            if classes[class_id] in ["car", "truck", "bus"]:
                # Calculate distance to object
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                objects = car_cascade.detectMultiScale(gray, 1.1, 5)
                for (ox, oy, ow, oh) in objects:
                    cv2.rectangle(blended, (ox, oy), (ox+ow, oy+oh), (0, 255, 0), 2)
                    object_center = (ox + ow // 2, oy + oh // 2)
                    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
                    distance_pixels = calculate_distance(object_center, frame_center)
                    distance_inches = (ref_width * focal_length) / distance_pixels
                    distance_feet = round(distance_inches / 12, 2)
                    cv2.putText(blended, f"{distance_feet} feet", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                   # If the object is within 10 feet, display an alert message on the screen
                    if distance_feet < 10:
                        alert_message = "{} is {} FT".format(class_name, round(distance_feet, 2))
                        cv2.putText(blended, alert_message, (int(frame.shape[1] / 2 - 150), int(frame.shape[0] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        cv2.rectangle(blended, (int(frame.shape[1] / 2 - 150), int(frame.shape[0] / 2 - 75)), (int(frame.shape[1] / 2 + 150), int(frame.shape[0] / 2 + 75)), (0, 0, 255), 2)

                    #

        cv2.waitKey(1)
    alert_sound.release()
    cv2.destroyAllWindows()
