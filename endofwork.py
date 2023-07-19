import cv2
import numpy as np
import math
from gui_buttons import Buttons

selected_object = None


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

# Load YOLOv4-tiny model
net = cv2.dnn.readNet("dnn_model\yolov4-tiny.weights", "dnn_model\yolov4-tiny.cfg")
classes = []
with open("dnn_model\classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
output_layers = net.getUnconnectedOutLayersNames()
input_size = (320, 320)

# Initialize video stream
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

# Load reference image and calculate focal length
ref_image = cv2.imread("INBAY.png")
gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
car_cascade = cv2.CascadeClassifier("cars1.xml")
cars = car_cascade.detectMultiScale(gray_ref, 1.1, 5)
ref_distance = 120  # 10 feet = 120 inches
ref_width = 48  # 4 feet = 48 inches
focal_length = (ref_distance * ref_width) / cars[0][2]

# Function to handle mouse clicks
def click_button(event, x, y, flags, params):
    global selected_object
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_object = None
        for button_name, button_rect in button.buttons.items():
            if button_rect.collidepoint(x, y):
                selected_object = button_name

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

# Main loop
while True:
    # Read frame from video stream
    ret, frame = cap.read()
    
    active_buttons = button.active_buttons_list()
    
       # Process frame with YOLOv4-tiny model
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, input_size, swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)
    class_ids, confidences, boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, width, height = map(int, detection[:4] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                x, y = center_x - width // 2, center_y - height // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])

    # Draw boxes and calculate distances
    for i, (class_id, confidence, box) in enumerate(zip(class_ids, confidences, boxes)):
        x, y, w, h = box
        label = f"{classes[class_id]}: {confidence:.2f}"
        color = colors[class_id]

        if selected_object is None or classes[class_id] == selected_object:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if classes[class_id] in ["car", "truck", "bus"]:
                # Calculate distance to object
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                objects = car_cascade.detectMultiScale(gray, 1.1, 5)
                for (ox, oy, ow, oh) in objects:
                    cv2.rectangle(frame, (ox, oy), (ox+ow, oy+oh), (0, 255, 0), 2)
                    object_center = (ox + ow // 2, oy + oh // 2)
                    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
                    distance_pixels = calculate_distance(object_center, frame_center)
                    distance_inches = (ref_width * focal_length) / distance_pixels
                    distance_feet = round(distance_inches / 12, 2)
                    cv2.putText(frame, f"{distance_feet} feet", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display buttons and frame
    button.display_buttons(frame)
    cv2.imshow("Frame", frame)

    # Handle key presses
    key = cv2.waitKey(1)
    if key == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
