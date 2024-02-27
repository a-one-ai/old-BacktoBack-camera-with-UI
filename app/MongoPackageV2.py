#Importing Packages
from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import pytz
import math
from pymongo.errors import OperationFailure
from time import sleep

#Assign Client Connection and Database as Global
global client , db


#Connect to the default local MongoDB instance
client = MongoClient('mongodb://localhost:27017/')
#Connect to Databse
db = client.CameraProject2


#Checking Existing Documents 
def check_existing_document(existing_collection, query):
    return existing_collection.find_one(query) is not None


#Finding Existing Documents 
def find_existing_document(existing_collection, query):
    return existing_collection.find_one(query)


#Updating  Existing Documents 
def update_existing_document(existing_collection, query, update_data):
    # Update an existing document based on the query with the provided update_data
    result = existing_collection.update_one(query, {'$set': update_data})
    return result.modified_count

#________________________________________________________


#Insert Camera Information
def insert_camera_info(CameraName, SourceType, Source, x, y):
    
    # Connect to collection
    existing_collection = db['CameraInfo']

    # Get current UTC time
    current_time_utc = datetime.utcnow()

    # Convert UTC time to Egypt timezone
    egypt_tz = pytz.timezone('Africa/Cairo')
    current_time_egypt = current_time_utc.replace(tzinfo=pytz.utc).astimezone(egypt_tz)

    # Format the current time in YYYY-MM-DD format
    date_str = current_time_egypt.strftime("%Y-%m-%d")

    # Prepare data for insertion
    data = {
        'Camera Name': CameraName,
        'Source Type': SourceType,
    }

    # Add specific fields based on the SourceType
    if SourceType == 'WEBCAM':
        data['Port'] = int(Source)
    elif SourceType in ['RTSP', 'URL']:
        data['Link'] = Source

    # Add location coordinates, status, and timestamp information
    data['Location Coordinates'] = {'x': x, 'y': y}
    data['Status'] = 'OFF'
    data['Insertion Timestamp'] = current_time_egypt  # Store the timestamp as a datetime object
    data['Insertion Date'] = date_str

    # Check if the document with the given Camera Name already exists
    query = {'Camera Name': CameraName}
    if check_existing_document(existing_collection, query):
        print('This Camera Name Already Exists')
    else:
        # Insert the document into the collection
        inserted_document = existing_collection.insert_one(data)
        # Print a success message with the inserted document ID
        print('Inserted Successfully with ID:', inserted_document.inserted_id)
        # Return the inserted document (optional, depending on your needs)
        return inserted_document


#insert_camera_info('Web','WEBCAM',0,1,2)


#_________________________________________________________________



#Inserting Model Information
def insert_model_info(CameraName, ModelName, Label, FramePath):
    
    # Determine the appropriate collection based on the ModelName
    if ModelName == 'violence':
        existing_collection = db['ModelViolenceData']
    elif ModelName == 'vehicle':
        existing_collection = db['ModelVehicleData']
    elif ModelName == 'crowdedDensity':
        existing_collection = db['ModelDensityData']
    elif ModelName == 'crossingBorder':
        existing_collection = db['ModelCountingData']
    elif ModelName == 'crowded':
        existing_collection = db['ModelCrowdedData']
    elif ModelName == 'Gender':
        existing_collection = db['ModelGenderData']

    # Get the current date and time in UTC
    current_time_utc = datetime.utcnow()

    # Define the timezone for Egypt (Eastern European Time)
    egypt_tz = pytz.timezone('Africa/Cairo')

    # Convert UTC time to Egypt timezone
    current_time_egypt = current_time_utc.replace(tzinfo=pytz.utc).astimezone(egypt_tz)


    # Extract the date component
    date_str = current_time_egypt.strftime("%Y-%m-%d")

    # Prepare data for insertion
    data = {
        'Model Name': ModelName,
    }

    # Check if the camera with the given name exists
    query = {'Camera Name': CameraName}
    camera_collection = db['CameraInfo']
    if check_existing_document(camera_collection, query):
        print('Camera Found')
        
        # Update the camera status to 'ON'
        update_existing_document(camera_collection, query, {'Status': 'ON'})
        
        # Retrieve camera data
        camera_data = find_existing_document(camera_collection, query)
        
        # Add camera information to the data
        data['Camera Info'] = camera_data

    else:
        print('Camera Not Added in Camera Collection')
        return 'Camera Not Added in Camera Collection'

    # Check the type of Label and set Count or Label accordingly
    if isinstance(Label, int) or ModelName in ['vehicle', 'crowdedDensity', 'crossingBorder', 'crowded']:
        data['Count'] = Label
    elif isinstance(Label, str) or ModelName not in ['vehicle', 'crowdedDensity', 'crossingBorder', 'crowded']:
        data['Label'] = Label

    # Add Frame Path, Timestamp, and Date information
    data['Frame Path'] = FramePath
    data['Timestamp'] = current_time_egypt
    data['Date'] = date_str

    # Insert the document into the collection
    inserted_document = existing_collection.insert_one(data)
    
    # Print a success message with the inserted document ID
    print(f'Inserted Successfully with ID in {ModelName} Collection: {inserted_document.inserted_id}')
    return inserted_document

#insert_model_info('Web','crowdedDensity',80,'352.png')



#_________________________________________________________________




#Returning all camera names in DB 
def finding_camera_names():
    db = client.CameraProject2    
    existing_collection = db['CameraInfo']
    cursor = existing_collection.find({})
    camera_names = [document['Camera Name'] for document in cursor]
    return camera_names


#_______________________________________________________
# #Filter by Data and Get Average of Count in Form of Time Range
# def date_filter_aggerigates_html(CameraName, ModelName,TargetDate) :
    
#     # Determine the appropriate collection based on the ModelName
#     if ModelName == 'violence':
#         existing_collection = db['ModelViolenceData']
#     elif ModelName == 'vehicle':
#         existing_collection = db['ModelVehicleData']
#     elif ModelName == 'crowdedDensity':
#         existing_collection = db['ModelDensityData']
#     elif ModelName == 'crossingBorder':
#         existing_collection = db['ModelCountingData']
#     elif ModelName == 'crowded':
#         existing_collection = db['ModelCrowdedData']
        
#     query = {'Camera Info.Camera Name': CameraName}
    
#     if check_existing_document(existing_collection, query):
#         print(f'Camera Found in {ModelName} Collection')
#         # Create the aggregation pipeline
#         pipeline = [
#             {
#                 "$match": {"Date": TargetDate}
#             },
#             {
#                 "$group": {
#                     "_id": {"$hour": "$Timestamp"},
#                     "count": {"$avg": "$Count"}
#                 }
#             },
#             {
#                 "$project": {
#                     "Hour": "$_id",
#                     "Count Average": "$count",
#                     "_id": 0
#                 }
#             },
#             {
#                 "$sort": {"Hour": 1}
#             }
#         ]                                                

#         result = list(existing_collection.aggregate(pipeline))  
#         # Generate HTML table directly with spaces and formatting
#         html_table = (
#             "<table border='1'>"
#             "<tr><th style='text-align:left;'>Time Range</th>"
#             "<th style='text-align:center;'>Count Average</th></tr>"
#         )

#         for item in result:
#             hour = item['Hour']
#             average_count = math.ceil(item['Count Average'])

#             # Determine AM or PM based on the hour
#             am_pm = "PM" if (hour+2) >= 12 else "AM"
#             formatted_hour = (hour+2) if (hour+2) <= 12 else (hour+2) - 12

#             # Format time range
#             time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"

#             # Add a row to the HTML table
#             html_table += (
#                 f"<tr><td style='text-align:left;'>{time_range}</td>"
#                 f"<td style='text-align:center;'>{average_count}</td></tr>"
#             )

#         # Close the HTML table
#         html_table += "</table>"

#         return html_table    
#     else :
#         return f'Camera not Found in {ModelName} Collection'
                       
#TargetDate = '2024-02-01'              
#aggerigates = date_filter_aggerigates_html('Density_Cam','crowdedDensity',TargetDate)
#print(aggerigates)



# #Filter by Data and Get Average of Count in Form of Time Range
# def date_filter_aggerigates_df(CameraName, ModelName,day , month,year) :

#     if int(month) < 10:
#         month = '0' + str(month)
#     if int(day) < 10:
#         day = '0' + str(day)

#     TargetDate = str(year) + '-' + str(month) + '-' + str(day)    

#     # Determine the appropriate collection based on the ModelName
#     if ModelName == 'violence':
#         existing_collection = db['ModelViolenceData']
#     elif ModelName == 'vehicle':
#         existing_collection = db['ModelVehicleData']
#     elif ModelName == 'crowdedDensity':
#         existing_collection = db['ModelDensityData']
#     elif ModelName == 'crossingBorder':
#         existing_collection = db['ModelCountingData']
#     elif ModelName == 'crowded':
#         existing_collection = db['ModelCrowdedData']
        
#     query = {'Camera Info.Camera Name': CameraName}
    
#     if check_existing_document(existing_collection, query):
#         print(f'{CameraName} Camera Found in {ModelName} Collection')
#         # Create the aggregation pipeline
#         pipeline = [
#             {
#                 "$match": {"Date": TargetDate}
#             },
#             {
#                 "$group": {
#                     "_id": {"$hour": "$Timestamp"},
#                     "count": {"$avg": "$Count"}
#                 }
#             },
#             {
#                 "$project": {
#                     "Hour": "$_id",
#                     "Count Average": "$count",
#                     "_id": 0
#                 }
#             },
#             {
#                 "$sort": {"Hour": 1}
#             }
#         ]                                                

#         result = list(existing_collection.aggregate(pipeline))  
#         # Generate HTML table directly with spaces and formatting
#     if result:
        
#         data = []

#         for item in result:
#             hour = item['Hour']
#             average_count = math.ceil(item['Count Average'])

#             # Determine AM or PM based on the hour
#             am_pm = "PM" if (hour + 2) >= 12 else "AM"
#             formatted_hour = (hour + 2) if (hour + 2) <= 12 else (hour + 2) - 12

#             # Format time range
#             time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"

#             data.append({'Time Range': time_range, 'Count Average': average_count})
            
#         #Get Pandas DataFrame
#         df = pd.DataFrame(data)
#         return df

#     else:
#         return f'{CameraName} Camera not Found in {ModelName} Collection'
    

import pandas as pd

#Filter by Data and Get Average of Count in Form of Time Range
def date_filter_aggerigates_df(CameraName, ModelName, day, month, year):
    if int(month) < 10:
        month = '0' + str(month)
    if int(day) < 10:
        day = '0' + str(day)

    TargetDate = str(year) + '-' + str(month) + '-' + str(day)    

    # Determine the appropriate collection based on the ModelName
    if ModelName == 'violence':
        existing_collection = db['ModelViolenceData']
    elif ModelName == 'vehicle':
        existing_collection = db['ModelVehicleData']
    elif ModelName == 'crowdedDensity':
        existing_collection = db['ModelDensityData']
    elif ModelName == 'crossingBorder':
        existing_collection = db['ModelCountingData']
    elif ModelName == 'crowded':
        existing_collection = db['ModelCrowdedData']
    elif ModelName == 'Gender':
        existing_collection = db['ModelGenderData']
        
    query = {'Camera Info.Camera Name': CameraName}
    
    if check_existing_document(existing_collection, query):
        print(f'{CameraName} Camera Found in {ModelName} Collection')
        # Create the aggregation pipeline
        pipeline = [
            {
                "$match": {"Date": TargetDate}
            },
            {
                "$group": {
                    "_id": {"$hour": "$Timestamp"},
                    "count": {"$avg": "$Count"}
                }
            },
            {
                "$project": {
                    "Hour": "$_id",
                    "Count Average": "$count",
                    "_id": 0
                }
            },
            {
                "$sort": {"Hour": 1}
            }
        ]                                                

        result = list(existing_collection.aggregate(pipeline))  
        
        if result:
            data = []

            for item in result:
                hour = item['Hour']
                average_count = math.ceil(item['Count Average'])

                # Determine AM or PM based on the hour
                am_pm = "PM" if (hour + 2) >= 12 else "AM"
                formatted_hour = (hour + 2) if (hour + 2) <= 12 else (hour + 2) - 12

                # Format time range
                time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"

                data.append({'Time Range': time_range, 'Count Average': average_count})

            #Get Pandas DataFrame
            df = pd.DataFrame(data)
            return df
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no data found
    else:
        return pd.DataFrame()  # Return an empty DataFrame if camera not found
