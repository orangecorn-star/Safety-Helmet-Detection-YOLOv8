#coding:utf-8
import cv2
from ultralytics import YOLO

# Assuming you want to use the YOLOv8 enhanced model.
path = r'runs\detect\train_yolov8_enhanced\weights\best.pt'

# Make sure you have a folder named TestFiles with a video file (e.g., 1.mp4) in your project root.
video_path = "TestFiles/1.mp4"

# Load the YOLOv8 model
model = YOLO(path)
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference (Enhanced)", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()