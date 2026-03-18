from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    results = model(frame)
    cv2.imshow("Detection", results[0].plot())
    if cv2.waitKey(1) == 27:
        break

cap.release() #will add a funtion to exit the process
cv2.destroyAllWindows()
