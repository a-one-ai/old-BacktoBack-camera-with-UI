import cv2
from ultralytics import YOLO
import numpy as np
import math
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
torch.cuda.set_device(0) 
num_cpu_cores = torch.multiprocessing.cpu_count()
#torch.set_num_threads(4)  # Set the number of threads for PyTorch


# Set GPU device if available
if torch.cuda.is_available():
    torch.cuda.set_device(device)
else :
        print('no gpu')

num_cpu_cores = torch.multiprocessing.cpu_count()

# Ensure that the YOLO model is configured for GPU
model = YOLO("D:\Gender Detection\\runs\detect\\train2\weights\\best.pt")

def detect_GENDER(frame):
    try:
        # Assign image to model
        results = model(frame, stream=True)

        # Getting bbox, confidence, and class names information to work with
        # Assign image to model to detect people and get boxes
        for info in results:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                # Add box if confidence of detection more than or equal to 40% and count objects
                if confidence >= 40 :
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 4)
                    # Display label
                    label = model.names[int(Class)]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    font_thickness = 2
                    text_color = (0, 120, 255)
                    cv2.putText(frame, f"{label}: {confidence}%", (x1, y1 - 10),
                                font, font_scale, text_color, font_thickness)
                    
        return frame , label

    except Exception as e:
        print(f'>> Error: {str(e)}')
        return frame 

    
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame =  detect_GENDER(frame)
    cv2.imshow(f'Image', frame)
    
    # Check for 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()