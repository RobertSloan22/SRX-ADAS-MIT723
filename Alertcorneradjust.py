import cv2
from gui_buttons import Buttons
import numpy as np
import math
import radar

# ... (previous code remains the same)

# Function to define region of interest (ROI)
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Function to draw lane lines on the frame
def draw_lines(img, lines, color=[0, 255, 0], thickness=3):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img = np.copy(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img

# ... (previous code remains the same)

while True:
    ret, frame = cap.read()
    
    radar.draw_radar(frame)

    # ... (previous code remains the same)

    # Lane line detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    canny_frame = cv2.Canny(blur_frame, 50, 150)

    height = frame.shape[0]
    width = frame.shape[1]
    roi_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    roi_frame = region_of_interest(canny_frame, np.array([roi_vertices], np.int32))
    
    lines = cv2.HoughLinesP(roi_frame, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]), minLineLength=40, maxLineGap=25)
    frame_with_lines = draw_lines(frame, lines)

    # ... (previous code remains the same)

    button.display_buttons(frame_with_lines)

    cv2.imshow("Frame", frame_with_lines)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
