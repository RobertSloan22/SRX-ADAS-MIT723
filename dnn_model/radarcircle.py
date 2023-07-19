import cv2
import numpy as np
import math

def draw_radar(frame, cap, circle_spacing=50, circle_speed=5):
    """
    Draws a radar-like graphic on the given frame.
    """
    circles = []
    circle_radius = 0
    circle_counter = 0
    circle_interval = 5  # Generate a new circle every 5 cycles
    line_x = 0
    line_speed = 10
    line_length = 50

    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / circle_spacing)):
        center_x = (i + 1) * circle_spacing
        circle_center = (center_x, cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        circles.append((circle_center, 0))

    new_circles = []
    for circle in circles:
        circle_radius += circle_speed
        if circle_radius > int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)):
            continue
        else:
            circle_center = (circle[0][0], int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - circle_radius)
            cv2.circle(frame, circle_center, circle_radius, (255, 0, 0), thickness=2)
            new_circles.append((circle_center, circle_radius))

    if circle_counter % circle_interval == 0:
        new_circle_center = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        new_circles.append((new_circle_center, 0))
    circle_counter += 1

    line_x += line_speed
    if line_x > cap.get(cv2.CAP_PROP_FRAME_WIDTH):
        line_x = 0
    cv2.line(frame, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), (line_x, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - line_length), (0, 255, 0), thickness=2)

    circles = new_circles
    return frame, circles
while True:
    ret, frame = cap.read()

    # Draw radar
    frame, circles = radar.draw_radar(frame, cap)

    # Process buttons
    active_buttons = button.active_buttons_list()

    # Detect objects and draw bounding boxes and lines
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
                        cv2.putText(frame, alert_message, (int(frame.shape[1] / 2 - 150), int(frame.shape[0] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        cv2.rectangle(frame, (int(frame.shape[1] / 2 - 150), int(frame.shape[0] / 2 - 75)), (int(frame.shape[1] / 2 + 150), int(frame.shape[0] / 2 + 75)), (0, 0, 255), 2)

   # Update and draw circles
    new_circles = []
    for circle in circles:
        circle_radius += circle_speed
        if circle_radius > int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2):
            continue
        else:
            circle_center = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - circle_radius))
            cv2.circle(frame, circle_center, circle_radius, (255, 0, 0), thickness=2)
            new_circles.append((circle_center, circle_radius))

    # Add new circles
    if circle_counter % circle_interval == 0:
        new_circle_center = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        new_circles.append((new_circle_center, 0))
    circle_counter += 1

    # Draw lines
    line_x += line_speed
    if line_x > cap.get(cv2.CAP_PROP_FRAME_WIDTH):
        line_x = 0
    cv2.line(frame, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), (line_x, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - line_length), (0, 255, 0), thickness=2)

    # Display buttons and frame
    button.display_buttons(frame)
    cv2.imshow("Frame", frame)

    # Check for exit
    key = cv2.waitKey(1)
    if key == 27:
        break

    # Update circle list
    circles = new_circles

cap.release()
cv2.destroyAllWindows()
