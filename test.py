from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(0)

frame_count = 0
last_a = None 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640,480))
    frame_count += 1

    if frame_count % 3 == 0:
        results = model(frame, imgsz=320, conf=0.4, verbose=False)
        last_a = results[0].plot()

    if last_a is not None:
        cv2.imshow("YOLOv8 Detection", last_a)
    else:
        cv2.imshow("YOLOv8 Detection", frame)


    if cv2.waitKey(1) == 27: #ESC
        break
        
cap.realease()
cv2.destroyAllWindows()
