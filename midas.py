import cv2
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from midas.midas_net import MidasNet
from gui_buttons import Buttons

cap = cv2.VideoCapture(0)

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

model = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model.setInputParams(size=(320, 320), scale=1 / 255)

classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# Load the MiDaS model
midas_model = MidasNet()
midas_model.load_state_dict(torch.load("path/to/weights/file/model-f6b98070.pt"))
midas_model.eval()

# Define the image transforms
transform = Compose([
    Resize(384),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])

def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)

def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Frame", click_button)
while True:
    ret, frame = cap.read()

    active_buttons = button.active_buttons_list()

    # Detect objects using the YOLOv4-tiny model
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (320, 320), swapRB=True, crop=False)
    model.setInput(blob)
    detections = model.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            class_id = int(detections[0, 0, i, 1])
            class_name = classes[class_id]
            if class_name in active_buttons:
                color = colors[class_id]
                x, y, w, h = int(detections[0, 0, i, 3] * frame.shape[1]), int(detections[0, 0, i, 4] * frame.shape[0]), \
                             int(detections[0, 0, i, 5] * frame.shape[1]), int(detections[0, 0, i, 6] * frame.shape[0])

                # Calculate the distance using the MiDaS model
                center = (x + w // 2, y + h // 2)
                input_image = transform(frame).unsqueeze(0)
                with torch.no_grad():
                    depth = midas_model(input_image)
                depth = depth.squeeze().cpu().numpy()
                distance_in_meters = depth[center[1], center[0]]
                distance_in_feet = distance_in_meters * 3.28084

                # If the object is within 10 feet, display an alert message on the screen
                if distance_in_feet < 10:
                    alert_message = "{} is {} feet away".format(class_name, round(distance_in_feet, 2))
                    cv2.putText(frame, alert_message, (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

    button.display_buttons(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

