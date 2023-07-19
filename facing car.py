import cv2

# variables
# distance from camera to object(face) measured
KNOWN_DISTANCE = 76.2  # centimeter
# width of face in the real world or Object Plane
KNOWN_WIDTH = 14.3  # centimeter
# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX
cap = cv2.VideoCapture(0)

# face detector object
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
car_detector = cv2.CascadeClassifier("cars1.xml") # car detector object

# reference image for finding the focal length
ref_image = cv2.imread("IMG-4746.jpg")
ref_image_car_width = 4 * ref_image.shape[1] / 10 # assuming the car in the reference image occupies 40% of the width
focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_car_width)
print(focal_length_found)
cv2.imshow("ref_image", ref_image)

while True:
    _, frame = cap.read()

    # calling face_data function
    face_width_in_frame = face_data(frame)
    # detecting cars in the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in cars:
        # calculate the distance to the car using its width in the frame
        car_width_in_frame = w
        Distance = distance_finder(focal_length_found, 4, car_width_in_frame)
        # draw the bounding box around the car and display the distance
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        cv2.putText(frame, f"Car Distance = {round(Distance,2)} feet", (x, y - 10), fonts, 0.5, WHITE, 2)
        
    if face_width_in_frame != 0:
        # calculate the distance to the face using its width in the frame
        Distance = distance_finder(focal_length_found, KNOWN_WIDTH, face_width_in_frame)
        # draw the bounding box around the face and display the distance
        cv2.rectangle(frame, (10, 10), (200, 200), RED, 2)
        cv2.putText(frame, f"Face Distance = {round(Distance,2)} CM", (50, 50), fonts, 1, WHITE, 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
