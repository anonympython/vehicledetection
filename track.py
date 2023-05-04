from ultralytics import YOLO
import cv2
from IPython import display

from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

from supervision.video.dataclasses import VideoInfo
from supervision.video.sink import VideoSink
from supervision.video.source import get_video_frames_generator

from tqdm.notebook import tqdm


MODEL_PATH = "static/model/best.pt"

model = YOLO(MODEL_PATH)
model.fuse()


LINE_START = Point(50, 1500)
LINE_END = Point(3840-50, 1500)

CLASS_NAMES_DICT = model.model.names

CLASS_ID = [0,1,2,3,4,5,6]
# CLASS_ID = [0,1]

print(model.model.names)

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

from typing import List

import numpy as np


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections, 
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    
    tracker_ids = [None] * len(detections)
    
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids



box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1)


def process(SOURCE_VIDEO_PATH,segment):

    
    data = {}
    temp = []

    byte_tracker = BYTETracker(BYTETrackerArgs())
    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    # create frame generator
    generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
    # create LineCounter instance
    line_counter = LineCounter(start=LINE_START, end=LINE_END)
    # create instance of BoxAnnotator and LineCounterAnnotator
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=1, text_scale=1)
    line_annotator = LineCounterAnnotator(thickness=1, text_thickness=1, text_scale=1)

    if(segment==0):  
        TARGET_VIDEO_PATH  = "Detected_processed_video_" + SOURCE_VIDEO_PATH 
    else:
        TARGET_VIDEO_PATH  = "Segmentated_processed_video_" + SOURCE_VIDEO_PATH  


    # open target video file
    with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        # loop over video frames
        for frame in generator:
            # model prediction on single frame and conversion to supervision Detections
            results = model(frame)
            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )
            # filtering out detections with unwanted classes
            mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # tracking detections
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)
            # filtering out detections without trackers
            mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # format custom labels

            
            labels = [
                f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]
            
            if(segment == 0):
                frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
            else:
                frame = results[0].plot()
            
            sink.write_frame(frame)


            # val -> [[xyxy], confidence, class_id, track_id]
            for val in detections:
                if(data.__contains__(val[3])):
                    if(data[val[3]]['confidence']< val[1]):
                        data[val[3]]['confidence'] = val[1]
                        data[val[3]]['class_type'] = CLASS_NAMES_DICT[val[2]]
                else:
                    data.__setitem__(val[3] , {'confidence' : val[1], 'class_type' : CLASS_NAMES_DICT[int(val[2])]})
                temp.append(val)

    return data,temp
            

def get_count(data):
    min_confidence = 0.8
    vehicle_count = 0
    vehicle = {}

    for entry in data.values():
        if(entry['confidence']>=min_confidence):
            if(entry['class_type']!="Axle"):
                vehicle_count+=1
                if(vehicle.__contains__(entry['class_type'])):
                    vehicle[entry['class_type']]+=1
                else:
                    vehicle[entry['class_type']]=1

    return vehicle_count,vehicle

def get_prediction_from_frame(frame,byte_tracker,data,temp):
    results = model(frame)
    detections = Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    )
    # filtering out detections with unwanted classes
    mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
    detections.filter(mask=mask, inplace=True)
    # tracking detections
    tracks = byte_tracker.update(
        output_results=detections2boxes(detections=detections),
        img_info=frame.shape,
        img_size=frame.shape
    )
    tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
    detections.tracker_id = np.array(tracker_id)
    # filtering out detections without trackers
    mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
    detections.filter(mask=mask, inplace=True)
    # format custom labels

    
    labels = [
        f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections
    ]
    # annotate and display frame
    # frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    frame = results[0].plot()


    # val -> [[xyxy], confidence, class_id, track_id]
    for val in detections:
        if(data.__contains__(val[3])):
            if(data[val[3]]['confidence']< val[1]):
                data[val[3]]['confidence'] = val[1]
                data[val[3]]['class_type'] = CLASS_NAMES_DICT[val[2]]
        else:
            data.__setitem__(val[3] , {'confidence' : val[1], 'class_type' : CLASS_NAMES_DICT[int(val[2])]})
        temp.append(val)

    return frame,byte_tracker,data,temp