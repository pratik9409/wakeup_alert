from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2


def sound_alarm(path):
    playsound.playsound(path)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])


    eyes = (A + B) / (2.0 * C)
    
    return eyes
 

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path for the video")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")

args = vars(ap.parse_args())
 

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48


COUNTER = 0
ALARM_ON = False


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


cap = cv2.VideoCapture(args['video'])
#address = "https://192.168.0.102:8080/video"
#cap.open(address)

#vs = VideoStream(src="cap").start()
#time.sleep(1.0)


while True:
    _,frame = cap.read()
    frame = cv2.resize(frame, None, fx = 1.2, fy = 1.2)
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    rects = detector(gray, 0)

    
    for rect in rects:
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        ear = (leftEAR + rightEAR) / 2.0
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 0), 1)
        
        
        if ear > EYE_AR_THRESH:
            COUNTER += 1
            
            if COUNTER <= EYE_AR_CONSEC_FRAMES:
           
                if not ALARM_ON:
                   ALARM_ON = True
                   
                   if args["alarm"] != "1":
                       t = Thread(target=sound_alarm,args=(args["alarm"],))
                       t.deamon = True
                       t.start()
               
                cv2.putText(frame, "Baby WakeUp ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		
        else:
            COUNTER = 0
            ALARM_ON = False
            
            #cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Frame", frame)
	
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
#vs.stop()
