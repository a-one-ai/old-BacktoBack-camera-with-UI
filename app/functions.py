from pytube import YouTube
from datetime import datetime
from projectModel import *
from pymongo import MongoClient
from MongoPackageV2 import *
import streamlink

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


global capture
def youtube(url):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(res="720p", progressive=True).first()
        if stream:
            video_url = stream.url
            return video_url
        else:
            print("No suitable stream found for the video.")
            return None
    except Exception as e:
        print(f"Error in capturing YouTube video: {e}")
        return None


def stream(url):
    streams = streamlink.streams(url)
    best_stream = streams["best"]
    return best_stream.url

def readSource(srcType, src):
    global capture
    try:
        if srcType == 'WEBCAM':
            # src = int(src)
            capture = cv2.VideoCapture(src , cv2.CAP_DSHOW)
        elif srcType == 'RTSP':
            # src = f'{src}'
            capture = cv2.VideoCapture(src)
        elif srcType == 'URL':
            # src = f'{src}'
            try:
                vsrc = youtube(src)
                capture = cv2.VideoCapture(vsrc)
            except Exception as e:
                print(f"Error in capturing YouTube video: {e}")
                vsrc = stream(src)
                capture = cv2.VideoCapture(vsrc)
    except Exception as e:
        print(f"Error in readSource: {e}")
        capture = None

    return capture


def videoFeed(cameraName, modelName):
    fps = 1
    delay = int(1000 / fps)
    modelName = f'{modelName}'
    query = {'Camera Name': cameraName}

    try :
        src = int(find_existing_document(db['CameraInfo'],query)['Port'])
    except :
        src = str(find_existing_document(db['CameraInfo'],query)['Link'])

    srcType = find_existing_document(db['CameraInfo'],query)['Source Type'] 

    print(src , srcType)
    cap = readSource(srcType, src)
    
    if cap is None:
        print("Error: Capture object is None.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if modelName == 'violence':
            path, res = violence(frame)
        elif modelName == 'vehicle':
            path, res = vehicleCounting(frame)
        elif modelName == 'crowdedDensity':
            path, res = crowdedDensity(frame)
        elif modelName == 'crossingBorder':
            _, path, res = crossingBorder(frame)
        elif modelName == 'crowded':
            path, res= crowded(frame)
        elif modelName == 'Gender':
            path , res = detect_GENDER(frame)

        yield path, res, cameraName, modelName

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


#_________________________________________________________
def processInsert(cameraName, modelName):
    
    generator = videoFeed(cameraName, modelName)

    for result in generator:
        # Unpack the first five values, capture the rest in a list
        path, res, cameraName, modelName, *extra_values = result

        # Perform any additional processing or insert into MongoDB
        data = insert_model_info(cameraName, modelName, res ,path)

        # If there are extra values, print them
        if extra_values:
            print("Extra values:", extra_values)