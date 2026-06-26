#coding:utf-8
import cv2
from ultralytics import YOLO

# 【修改点1】：将路径改为你实际训练出的模型权重路径！
# 假设你想用 YOLOv8 增强版模型（请根据你的实际文件夹名字微调，比如 train_yolov8_enhanced 或 train_enhanced）
path = r'runs\detect\train_yolov8_enhanced\weights\best.pt'

# 【修改点2】：需要检测的视频地址。
# ⚠️警告：请确保你当前的工程目录下真的有一个叫 TestFiles 的文件夹，且里面有 1.mp4！
# 如果没有，请你在网上随便下个几秒钟的工地工人走动的视频，放到工程目录下，把这里改成具体的视频名字，比如 "test_video.mp4"
video_path = "TestFiles/1.mp4"

# Load the YOLOv8 model
model = YOLO(path)
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference (Enhanced)", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()