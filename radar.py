import cv2
import numpy as np
def draw_radar(frame):
    # create a black image of size (height, width, channels)
    height, width = frame.shape[:2]
    img = np.zeros((height, width, 3), np.uint8)

    # set up the parameters for the half moon circles
    circle_radius = 50
    circle_distance = 100
    num_circles = int((width - circle_radius) / (circle_radius + circle_distance))

    # set up the colors for the circles and their outlines
    circle_color = (0, 255, 0)
    outline_color = (0, 255, 255)

    # calculate the center of the first half moon
    center_x = int(width / 2)
    center_y = height - 10

    # draw the first half moon on the black image
    cv2.circle(img, (center_x - 2 * circle_radius, center_y), circle_radius, circle_color, -1)
    cv2.circle(img, (center_x + 2 * circle_radius, center_y), circle_radius, circle_color, -1)
    cv2.circle(img, (center_x, center_y - 2 * circle_radius), circle_radius, circle_color, -1)
    cv2.circle(img, (center_x - circle_radius, center_y - circle_radius), circle_radius, circle_color, -1)
    cv2.circle(img, (center_x + circle_radius, center_y - circle_radius), circle_radius, circle_color, -1)
    cv2.circle(img, (center_x, center_y - circle_radius), circle_radius, outline_color, 2)

    # draw the remaining circles on the black image
    for i in range(1, num_circles):
        x = center_x + i * (circle_radius + circle_distance)
        cv2.circle(img, (x, center_y), circle_radius, circle_color, -1)
        cv2.circle(img, (x, center_y), circle_radius, outline_color, 2)

    # overlay the camera image on top of the black image
    alpha = 0.5
    beta = 1.0 - alpha
    dst = cv2.addWeighted(frame, alpha, img, beta, 0.0)

    # display the result
    cv2.imshow("Radar", dst)

    # check for exit
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
