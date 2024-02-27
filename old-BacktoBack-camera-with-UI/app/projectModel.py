from modelsReqapp.violence.model import Model
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from modelsReqapp.density.src.models.CSRNet import CSRNet
from datetime import datetime
from ultralytics import YOLO
from modelsReqapp.yoloModels.tracker import Tracker
import pandas as pd
import math
import threading
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
#num_cpu_cores = torch.multiprocessing.cpu_count()
#torch.set_num_threads(4)  # Set the number of threads for PyTorch


# Set GPU device if available
if torch.cuda.is_available():
    print('gpu')
    torch.cuda.set_device(device)
else :
        print('no gpu')

num_cpu_cores = torch.multiprocessing.cpu_count()

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
            modelCrowd = YOLO('app/modelsReqapp/yoloModels/best_crowded.pt')

        if model is None :
             model = YOLO('app/modelsReqapp/yoloModels/yolov8s.pt')

        if modelV is None:
             modelV = Model()

        if modelDEns is None:
             modelDEns = CSRNet()

        if modelG is None:
             modelG = YOLO('app/modelsReqapp/yoloModels/gender.pt')


        
initialize_models()


my_file = open("app/modelsReqapp/yoloModels/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
tracker = Tracker()




#-----------gateCounting model -------------


#-----------density models -------------

modelDEns = CSRNet()
PATH = 'https://huggingface.co/muasifk/CSRNet/resolve/main/CSRNet.pth'
state_dict = torch.hub.load_state_dict_from_url(PATH, map_location=torch.device('cpu'))
modelDEns.load_state_dict(state_dict)
modelDEns.eval()
print('\n Model loaded successfully.. \n')

global x_density
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


#-----------crowded model -------------
global x_crowd
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




#-----------crossingBorder model -------------
global x_crossing
x_crossing = 0

def crossingBorder(frame):
    global x_crossing
    count = 0  
    results = model.predict(frame)
    # Print the type and content of 'results' for debugging
    print(f"Type of 'results': {type(results)}")
    print(f"Content of 'results': {results}")
    a = results[0].boxes.data
    try:
        a_gpu = torch.tensor(a).to("cuda:0") # Move to GPU
        px = pd.DataFrame(a_gpu.cpu().numpy()).astype("float") # Convert to NumPy on CPU
        print('gpu')
    except:
        px = pd.DataFrame(a).astype("float")
        print('cpu')

    bbox_list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            bbox_list.append([x1, y1, x2, y2])
            count += 1  

    bbox_id = tracker.update(bbox_list)
    for bbox in bbox_id:
        x3, y3, x4, y4, d = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)

    cv2.putText(frame, f'Count: {count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('dd' , frame)
    x_crossing += 1
    path = f'app/output/crossing/figure{x_crossing}.jpg'
    cv2.imwrite(path, frame)
    return frame , path, count



#-----------vehicleCounting models -------------
x_vehicle = 0
def vehicleCounting(frame):
    count = 0  

    results = model.predict(frame)
    a = results[0].boxes.data
    try:
        a_gpu = torch.tensor(a).to("cuda:0") # Move to GPU
        px = pd.DataFrame(a_gpu.cpu().numpy()).astype("float") # Convert to NumPy on CPU
        print('gpu')
    except:
        px = pd.DataFrame(a).astype("float")
        print('cpu')


    l = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        # Detect vehicles 
        if 'car' in c or 'truck' in c or 'bus' in c or 'bicycle' in c or 'motorcycle' in c:
            list.append([x1, y1, x2, y2])
            count += 1  

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, d = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        # cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)

    cv2.putText(frame, f'Count: {count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    x_vehicle = x_vehicle + 1
    path = f'app/output/vehicle/figure{x_vehicle}.jpg'
    cv2.imwrite(path , frame)
    return path , count






#-----------violence model --------------
global x_violence
x_violence = 0
# modelV = Model()

def violence(frame):
    global x_violence
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictions = modelV.predict(image=RGBframe)
    label = predictions['label']
    if label in ['violence in office', 'fight on a street','street violence'] :
                label = 'Predicted Violence'
    cv2.putText(frame, f'This is a {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    x_violence = x_violence + 1
    path = ''
    if label == 'Predicted Violence':
        path = f'app/output/violence/figure{x_violence}.jpg'
        cv2.imwrite(path , frame)

    return path , label 



#-----------Accidents model --------------



#-----------Gender model ------------------
global x_gender
x_gender = 0
def detect_GENDER(frame):
    global x_gender
    try:
        # Assign image to model
        results = modelG(frame, stream=True)

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
                    label = modelG.names[int(Class)]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    font_thickness = 2
                    text_color = (0, 120, 255)
                    cv2.putText(frame, f"{label}: {confidence}%", (x1, y1 - 10),
                                font, font_scale, text_color, font_thickness)
                    
                    x_gender = x_gender + 1
                    path = ''
                    path = f'app/output/gender/figure{x_gender}.jpg'
                    cv2.imwrite(path , frame)
                    
        return path , label

    except Exception as e:
        print(f'>> Error: {str(e)}')
        return None , None

