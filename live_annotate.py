
import cv2
from ultralytics import YOLO 
from track import *




def generate_frames():
    camera = cv2.VideoCapture(0)
    byte_tracker = BYTETracker(BYTETrackerArgs())
    data = {}
    temp = []
    while(True):
        success,frame = camera.read()
        if not success:
            break
        else:
            frame,byte_tracker,data,temp = get_prediction_from_frame(frame,byte_tracker,data,temp)
            ret,buffer = cv2.imencode('.jpg',frame)
            frame = buffer.tobytes()
            vehicle_count,vehicle = get_count(data)
            print(vehicle_count,vehicle )
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

