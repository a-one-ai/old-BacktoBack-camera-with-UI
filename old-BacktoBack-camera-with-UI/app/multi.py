from projectModel import *
import cv2
from modelsReq.violence.model import Model
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from modelsReq.density.src.models.CSRNet import CSRNet
from datetime import datetime
from ultralytics import YOLO
from modelsReq.yoloModels.tracker import Tracker
import pandas as pd
import math
import threading



#-----------initialize for models------------
modelCrowd  = None
model= None
modelV  = None
modelDEns = None
modelG = None

model_lock = threading.Lock()
count = 0
def initialize_models():
    global count 
    count += 1
    print("Initializing models for the {} time".format(count))
    global modelCrowd, model , modelV , modelDEns , modelG
    with model_lock:
        if modelCrowd is None:
            modelCrowd = YOLO('app/modelsReq/yoloModels/best_crowded.pt')

        if model is None :
             model = YOLO('app/modelsReq/yoloModels/yolov8s.pt')

        if modelV is None:
             modelV = Model()

        if modelDEns is None:
             modelDEns = CSRNet()

        if modelG is None:
             modelG = YOLO('app/modelsReq/yoloModels/gender.pt')


        
initialize_models()


my_file = open("app/modelsReq/yoloModels/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
tracker = Tracker()


modelDEns = CSRNet()
PATH = 'https://huggingface.co/muasifk/CSRNet/resolve/main/CSRNet.pth'
state_dict = torch.hub.load_state_dict_from_url(PATH, map_location=torch.device('cpu'))
modelDEns.load_state_dict(state_dict)
modelDEns.eval()
print('\n Model loaded successfully.. \n')

x_density = 0 

def crowdedDensity(frame):
    global x_density

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255  # normalize image
    frame = torch.from_numpy(frame).permute(2, 0, 1)  # reshape to [c, w, h]

    # predict
    predict = modelDEns(frame.unsqueeze(0))
    count = predict.sum().item()

    # Plot the results using Matplotlib
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
    ax0.imshow(frame.permute(1, 2, 0))  # reshape back to [w, h, c]
    ax1.imshow(predict.squeeze().detach().numpy(), cmap='jet')
    ax0.set_title('People Count')
    ax1.set_title(f'People Count = {count:.0f}')
    ax0.axis("off")
    ax1.axis("off")
    plt.tight_layout()

    # Save the figure
    x_density = x_density + 1
    path = f'app/output/density/figure{x_density}.jpg'
    plt.savefig(path)  # Specify the desired output file path and format
    plt.close()  # Close the figure to release resources
    print('Figure saved successfully.')

    return path, count




x_crowd = 0
def crowded(frame):
    global x_crowd
    count = 0
    results = modelCrowd(frame,stream=True)

        # Getting bbox,confidence and class names informations to work with
        # Assign image to model to detect people and get boxes
    for info in results:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])     
                # Add box if confidence of detection more than or eqaul to 30% and count objects
                if confidence >= 40:
                    x1,y1,x2,y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0, 255, 0),2)
                    count +=1
                    
    cv2.putText(frame, f"Count : {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 5)
    x_crowd = x_crowd + 1
    path = f'app/output/crowded/figure{x_crowd}.jpg'
    cv2.imwrite(path , frame)
 
    return path ,  count



cap = cv2.VideoCapture('D:/sisi.mp4')

while True:
    ret , frame = cap.read()
    path , count = crowdedDensity(frame)
    p , c = crowded(frame)


    k = cv2.waitKey(0)
    if k ==27:
        break

cap.release()
cv2.destroyAllWindows()




