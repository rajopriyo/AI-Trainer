import pandas as pd
import cv2
import mediapipe as mp
import numpy as np
from landmarks import landmarks

cap=cv2.VideoCapture('Rajo.mp4')
def calculate_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)

    if(angle>180.0):
        angle=360-angle
    return angle

mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width)
print(height)
# data=pd.DataFrame(list())
listl1=[]
df=pd.read_csv("left.csv")
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    state=None
    count_deadlift=0
    try:
        while True:
            # print("Hello World")
            _,frame=cap.read()
            image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image.flags.writeable=True
            result=pose.process(image)
            image.flags.writeable=False
            image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            landmarks=result.pose_landmarks.landmark

            # print(landmarks)
            # templist=[]
            # for i in range (33):
            #     templist.append(landmarks[i].x)            
            #     templist.append(landmarks[i].y)            
            #     templist.append(landmarks[i].z)            
            #     templist.append(landmarks[i].visibility)    

            # listl1.append(templist)        

            # if df["x12"]==landmarks[12].x and df["y12"]==landmarks[12].y and df["x14"]==landmarks[14].x and df["y14"]==landmarks[14].y:
            #     print("Line Drawn")
            point={'x12':landmarks[11].x,'y12':landmarks[11].y,'x14':landmarks[13].x,'y14':landmarks[13].y}
            isContained=(df['x12']==point['x12']) & (df['y12']==point['y12']) & (df['x14']==point['x14']) & (df['y14']==point['y14'])
            is_point_found=any(isContained)

            # print(int((landmarks[11].x)*width))
            coorx=(int((landmarks[11].x)*width))
            coory=(int((landmarks[11].y)*height))
            if is_point_found:
                print(coorx)
                print(coory)
                
                # print()
                # print(int(landmarks[12].y*height))
                cv2.circle(image,(coorx,coory),color=(0,255,0),radius=20, thickness=-1)
                mp_drawing.draw_landmarks(image,result.pose_landmarks,mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),mp_drawing.DrawingSpec(color=(120,50,20),thickness=2,circle_radius=2))
                
                # print("Line Drawn")
            else:
                # print("Sorry")
                cv2.circle(image,(coorx,coory),color=(0,0,255),radius=20, thickness=-1)
                mp_drawing.draw_landmarks(image,result.pose_landmarks,mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),mp_drawing.DrawingSpec(color=(45,123,52),thickness=2,circle_radius=2))

            # print(df["x12"])

            cv2.imshow("Video Feed",image)
            if(cv2.waitKey(10) & 0xFF== ord('q')):
                break
    except:
        # print(df['x12'])
        pass