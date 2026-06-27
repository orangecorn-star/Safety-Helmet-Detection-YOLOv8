#coding:utf-8
from ultralytics import YOLO
import cv2
import os

# Path to the trained model
model = YOLO("./runs/detect/train/weights/best.pt")

# Path to the image to be detected (modifiable)
img_path = "./hard/4.jpg"

# Directory to save the results
save_dir = "./imageTest"

# Create the directory if it does not exist
os.makedirs(save_dir, exist_ok=True)

# Run inference on the image
results = model(img_path)
res = results[0].plot()

# Display the image (press any key to close the window)
cv2.imshow("YOLOv8 Detection", res)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image with bounding boxes
# Get the base filename and construct the save path
base_name = os.path.basename(img_path)
save_path = os.path.join(save_dir, base_name)
cv2.imwrite(save_path, res)
print(f"Detection result saved to: {save_path}")