from pytube import YouTube
from datetime import datetime
from projectModel import *
from pymongo import MongoClient
from MongoPackageV2 import *
import streamlink
import cv2
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
            fps = 30
        elif srcType == 'RTSP':
            # src = f'{src}'
            capture = cv2.VideoCapture(src)
            if 'rtsp' in src :
                fps = 30
                
            else :
                fps = capture.get(cv2.CAP_PROP_FPS)                
                    
        elif srcType == 'URL':
            # src = f'{src}'
            try:
                vsrc = youtube(src)
                capture = cv2.VideoCapture(vsrc)
                fps = capture.get(cv2.CAP_PROP_FPS)                
                
            except Exception as e:
                print(f"Error in capturing YouTube video: {e}")
                vsrc = stream(src)
                capture = cv2.VideoCapture(vsrc)
                fps = 30

                
        # Get the frame rate of the video
        print(fps)
        return capture , fps    
                           
    except Exception as e:
        print(f"Error in readSource: {e}")
        capture = None

        return capture





def videoFeed(cameraName, modelName):

    modelName = f'{modelName}'
    query = {'Camera Name': cameraName}

    try :
        src = int(find_existing_document(db['CameraInfo'],query)['Port'])
    except :
        src = str(find_existing_document(db['CameraInfo'],query)['Link'])

    srcType = find_existing_document(db['CameraInfo'],query)['Source Type'] 

    print(src , srcType)
    
    cap , fps = readSource(srcType, src)
    
    if cap is None:
        print("Error: Capture object is None.")
        return

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        count += 1 
        if  (count % int(fps) == 0) or (count == 1):    
            if count == 1 : 
                print(f'One Frame in {count} second')
                   
            #cv2.imshow('Frame',frame)
            current_sec = count/fps
            print(f'One Frame in {current_sec} second')
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

        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

#path, res, cameraName, modelName = videoFeed('CDA','crowded')



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
            
#processInsert('CDA','crowded')            