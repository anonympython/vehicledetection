# # import the opencv library
# import cv2
# from track import *


# def get_frame(vehicle,vehicle_count,min_confidence):
#     # define a video capture object
#     vid = cv2.VideoCapture(0)

#     while(True):
        
#         # Capture the video frame
#         # by frame
#         ret, frame = vid.read()

#         # Display the resulting frame
#         track_from_frame(frame,vehicle,vehicle_count,min_confidence)
        
#         # the 'q' button is set as the
#         # quitting button you may use any
#         # desired button of your choice
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # After the loop release the cap object
#     vid.release()
#     # Destroy all the windows
#     cv2.destroyAllWindows()


# def annotate_live_frame(vehicle,vehicle_count,min_confidence):
#     # define a video capture object
#     vid = cv2.VideoCapture(0)

#     while(True):
        
#         # Capture the video frame
#         # by frame
#         ret, frame = vid.read()

#         # Display the resulting frame
#         track_from_frame(frame,vehicle,vehicle_count,min_confidence)
        
#         # the 'q' button is set as the
#         # quitting button you may use any
#         # desired button of your choice
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # After the loop release the cap object
#     vid.release()
#     # Destroy all the windows
#     cv2.destroyAllWindows()