import cv2
import numpy as np

# Set up camera capture
cap = cv2.VideoCapture(0)

# Define colors
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

# Define circle parameters
circle_radius = 10
circle_thickness = 2
circle_speed = 1
circle_spacing = 50

# Define line parameters
line_thickness = 1
line_length = 50

# Initialize circle positions
circles = []
for i in range(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / circle_spacing)):
    circles.append({'x': i * circle_spacing, 'y': cap.get(cv2.CAP_PROP_FRAME_HEIGHT)})

# Main loop
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Draw sweeping radar line
    line_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * cv2.getTickCount() / cv2.getTickFrequency())
    if line_x >= cap.get(cv2.CAP_PROP_FRAME_WIDTH):
        line_x = 0
    cv2.line(frame, (line_x, 0), (line_x, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), RED, line_thickness)

    # Update and draw circles
    new_circles = []
    for circle in circles:
        circle_radius += circle_speed
        if circle_radius > int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2):
            circle_radius = 10
        circle_center = (int(circle['x']), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - circle_radius))
        cv2.circle(frame, circle_center, circle_radius, BLUE, circle_thickness)
        if circle_radius != 10:
            new_circles.append({'x': circle['x'], 'y': circle['y'] - circle_speed})
    circles = new_circles
    if len(circles) < int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / circle_spacing):
        circles.append({'x': len(circles) * circle_spacing, 'y': cap.get(cv2.CAP_PROP_FRAME_HEIGHT)})

    # Display the resulting frame
    cv2.imshow('Radar', frame)

    # Quit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
