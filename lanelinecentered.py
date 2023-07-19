import cv2
import numpy as np
import math
import radar
from gui_buttons import Buttons

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

lines = None

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

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Frame", click_button)
while True:
    ret, frame = cap.read()

    radar.draw_radar(frame)

    active_buttons = button.active_buttons_list()

    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        color = colors[class_id]

        if class_name in active_buttons:
            # Calculate the center and radius of the bounding circle
            center = (int(x + w/2), int(y + h/2))
            radius = int(math.sqrt((w/2)**2 + (h/2)**2))

            # Draw the bounding circle
            if class_name in active_buttons:
            # Calculate the center and radius of the bounding circle
                center = (int(x + w/2), int(y + h/2))
            radius = int(math.sqrt((w/2)**2 + (h/2)**2))
            # Draw the bounding circle
            cv2.circle(frame, center, radius, (255, 0, 0), thickness=2)
            # Apply Hough transform to detect lines
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)

            # Draw detected lines on frame
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

                # Detect lanes
                h, w, _ = frame.shape
                mask = np.zeros((h, w), np.uint8)
                region_of_interest_vertices = [(0, h), (w/2, h/2 + 50), (w, h)]

                cv2.fillPoly(mask, np.array([region_of_interest_vertices], np.int32), 255)
                cropped_image = cv2.bitwise_and(edges, edges, mask=mask)

                rho = 1  # distance resolution in pixels of the Hough grid
                theta = np.pi / 180  # angular resolution in radians of the Hough grid
                threshold = 15  # minimum number of votes (intersections in Hough grid cell)
                min_line_length = 60  # minimum number of pixels making up a line
                max_line_gap = 20  # maximum gap in pixels between connectable line segments
                lines = cv2.HoughLinesP(cropped_image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

                # Draw detected lanes on frame
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

    # Display buttons and frame
    button.display_buttons(frame)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

