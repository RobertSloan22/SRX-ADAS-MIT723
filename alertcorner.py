import cv2
from gui_buttons import Buttons
import numpy as np
import math
import radar



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
ref_image = cv2.imread("logitech.png")
gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
car_cascade = cv2.CascadeClassifier("cars1.xml")
cars = car_cascade.detectMultiScale(gray_ref, 1.1, 5)
ref_distance = 240  # 20 feet = 240 inches
ref_width = 78  # 6.5 feet = 78 inches
focal_length = (ref_distance * ref_width) / cars[0][2]
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


# Function to define region of interest (ROI)
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Function to draw lane lines and highlight the area between them
def draw_lines_and_highlight_area(img, left_line, right_line, color=[0, 255, 0], thickness=3):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img = np.copy(img)

    if left_line is not None and right_line is not None:
        # Draw left line
        cv2.line(line_img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), color, thickness)
        # Draw right line
        cv2.line(line_img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), color, thickness)
        # Fill the area between the lines
        pts = np.array([[left_line[0], left_line[1]], [left_line[2], left_line[3]], [right_line[2], right_line[3]], [right_line[0], right_line[1]]], dtype=np.int32)
        cv2.fillPoly(line_img, [pts], (150, 150, 150))

    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img

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
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    canny_frame = cv2.Canny(blur_frame, 50, 150)

    height = frame.shape[0]
    width = frame.shape[1]
    roi_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    roi_frame = region_of_interest(canny_frame, np.array([roi_vertices], np.int32))
    
    lines = cv2.HoughLinesP(roi_frame, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]), minLineLength=40, maxLineGap=25)
    if not np.isnan(left_lane).any() and not np.isnan(right_lane).any():
        frame_with_lines = draw_lines_and_highlight_area(frame, left_lane, right_lane)
    else:
        frame_with_lines = frame


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
            cv2.circle(frame, center, radius, (255, 0, 0), thickness=2)
            # Draw lines emanating from the center of the circle
            for angle in range(0, 360, 90):
                end_point_x = int(center[0] + radius * math.cos(math.radians(angle)))
                end_point_y = int(center[1] + radius * math.sin(math.radians(angle)))
                cv2.line(frame, center, (end_point_x, end_point_y), color, thickness=2)
            # Calculate distance to object
            if classes[class_id] in ["car", "truck", "bus"]:
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


                # If the object is within 10 feet, display an alert message on the screen
                if distance_feet < 10:
                    alert_message = "{} is {} FT".format(class_name, round(distance_feet, 2))
                    text_x = frame.shape[1] - 220
                    cv2.putText(frame, alert_message, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame, (text_x - 5, 5), (frame.shape[1] - 5, 105), (0, 0, 255), 2)
                # Lane line detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
                canny_frame = cv2.Canny(blur_frame, 50, 150)

                height = frame.shape[0]
                width = frame.shape[1]
                roi_vertices = [(0, height), (width / 2, height / 2), (width, height)]
                roi_frame = region_of_interest(canny_frame, np.array([roi_vertices], np.int32))

                lines = cv2.HoughLinesP(roi_frame, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]), minLineLength=40, maxLineGap=25)
                # Separate left and right lane lines
                left_lane_lines = []
                right_lane_lines = []

                if lines is not None:
                    for line in lines:
                        for x1, y1, x2, y2 in line:
                            slope = (y2 - y1) / (x2 - x1)
                            if math.fabs(slope) < 0.5:
                                continue
                            if slope <= 0:
                                left_lane_lines.append((x1, y1, x2, y2, slope))
                            else:
                                right_lane_lines.append((x1, y1, x2, y2, slope))

                # Calculate average left and right lane lines
                left_lane = np.mean(left_lane_lines, axis=0, dtype=np.int32)
                right_lane = np.mean(right_lane_lines, axis=0, dtype=np.int32)

                                # Extrapolate lane lines to cover the desired area
                def extrapolate_line(line, y1, y2):
                                    x1, y1_src, x2, y2_src, _ = line
                                    slope, intercept = np.polyfit((x1, x2), (y1_src, y2_src), 1)
                                    x1_extrapolated = (y1 - intercept) / slope
                                    x2_extrapolated = (y2 - intercept) / slope
                                    return int(x1_extrapolated), y1, int(x2_extrapolated), y2

                                y1_extrapolated = int(height * 0.6)
                                y2_extrapolated = height

                                if not np.isnan(left_lane).any():
                                    left_lane = extrapolate_line(left_lane, y1_extrapolated, y2_extrapolated)

                                if not np.isnan(right_lane).any():
                                    right_lane = extrapolate_line(right_lane, y1_extrapolated, y2_extrapolated)

                                frame_with_lines = draw_lines_and_highlight_area(frame, left_lane, right_lane)

    button.display_buttons(frame)
    button.display_buttons(frame_with_lines)

    cv2.imshow("Frame", frame_with_lines)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
