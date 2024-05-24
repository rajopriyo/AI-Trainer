import pandas as pd
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from landmarks import landmarks

cap=cv2.VideoCapture('Rajo.mp4')
# st.title("Hello !! Wave Your Hand  ")
st.set_page_config(
    page_title="AI Trainer",
    page_icon="💪",
)
# import streamlit.components.v1 as components
st.markdown(
    """
    <a href="http://localhost:8502" >
        <button style=" background-color: green;
    border: 4px solid green;
    border-radius: 10px;
    color: #bcff96;
    box-shadDataSetCreation.py;
    padding: 10px 20px ; cursor: pointer;">
            Go to Home
        </button>
    </a>
""",
    unsafe_allow_html=True,
)


st.write(
    """
# Hammer Curl Exercise AI Traier�💪
"""
)

start_button_pressed=st.button("")


frame_placeholder=st.empty()

stop_button_pressed=st.button("Stop")
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
# data=pd.DataFrame(list())
listl1=[]
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    state=None
    count_deadlift=0
    try:
        while not stop_button_pressed:
            
               
                _,image=cap.read()
                
                if not _:
                    st.write("The video capture has ended.")        
                
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image.flags.writeable=True
                result=pose.process(image)
                image.flags.writeable=False
                image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                landmarks=result.pose_landmarks.landmark
                mp_drawing.draw_landmarks(image,result.pose_landmarks,mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),mp_drawing.DrawingSpec(color=(0,0,255),thickness=2,circle_radius=2))
                frame_placeholder.image(image,channels="BGR")

                # print(landmarks)
                templist=[]
                for i in range (33):
                    templist.append(landmarks[i].x)            
                    templist.append(landmarks[i].y)            
                    templist.append(landmarks[i].z)            
                    templist.append(landmarks[i].visibility)    

                listl1.append(templist)        



                # cv2.imshow("Video Feed",image)
                if(cv2.waitKey(10) & 0xFF== ord('q')) or start_button_pressed:
                    break

    
    except:
        df=pd.DataFrame(listl1,columns=['x1',
 'y1',
 'z1',
 'v1',
 'x2',
 'y2',
 'z2',
 'v2',
 'x3',
 'y3',
 'z3',
 'v3',
 'x4',
 'y4',
 'z4',
 'v4',
 'x5',
 'y5',
 'z5',
 'v5',
 'x6',
 'y6',
 'z6',
 'v6',
 'x7',
 'y7',
 'z7',
 'v7',
 'x8',
 'y8',
 'z8',
 'v8',
 'x9',
 'y9',
 'z9',
 'v9',
 'x10',
 'y10',
 'z10',
 'v10',
 'x11',
 'y11',
 'z11',
 'v11',
 'x12',
 'y12',
 'z12',
 'v12',
 'x13',
 'y13',
 'z13',
 'v13',
 'x14',
 'y14',
 'z14',
 'v14',
 'x15',
 'y15',
 'z15',
 'v15',
 'x16',
 'y16',
 'z16',
 'v16',
 'x17',
 'y17',
 'z17',
 'v17',
 'x18',
 'y18',
 'z18',
 'v18',
 'x19',
 'y19',
 'z19',
 'v19',
 'x20',
 'y20',
 'z20',
 'v20',
 'x21',
 'y21',
 'z21',
 'v21',
 'x22',
 'y22',
 'z22',
 'v22',
 'x23',
 'y23',
 'z23',
 'v23',
 'x24',
 'y24',
 'z24',
 'v24',
 'x25',
 'y25',
 'z25',
 'v25',
 'x26',
 'y26',
 'z26',
 'v26',
 'x27',
 'y27',
 'z27',
 'v27',
 'x28',
 'y28',
 'z28',
 'v28',
 'x29',
 'y29',
 'z29',
 'v29',
 'x30',
 'y30',
 'z30',
 'v30',
 'x31',
 'y31',
 'z31',
 'v31',
 'x32',
 'y32',
 'z32',
 'v32',
 'x33',
 'y33',
 'z33',
 'v33'])
        # columns=['State','x1','y1','z1','v1','x2','y2','z2','v2','x3','y3','z3','v3','x4','y4','z4','v4','x5','y5','z5','v5','x6','y6','z6','v6']
        df=df.sample(frac=1)
        df.to_csv('data.csv',index=False)
    
cap.release()

try:
            dataFrame=pd.read_csv("data.csv")
            df=dataFrame[['x12','y12','x14','y14']]
            # colums_to_norm=['x12','y12','x14','y14']
            # scalar=MinMaxScaler()
            # df[colums_to_norm]=scalar.fit_transform(df[colums_to_norm])
            df.to_csv("Cleaneddata.csv",index=False)
             
except:
    exit()


try:
    clean_data=pd.read_csv('Cleaneddata.csv')
    X=clean_data.iloc[:,[0,1,2,3]].values
    kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
    Y=kmeans.fit_predict(X)
    # centroid_coordinates = kmeans.cluster_centers_
    centroids = kmeans.cluster_centers_
    X = clean_data.iloc[:, [0, 1, 2, 3]].values
    centroids1 = np.array([[0.26364887, 0.60247512, 0.3151196, 0.63653246],
                      [0.57609079, 0.28954591, 0.52803724, 0.30553756],
                      [0.58628793, 0.68121155, 0.58474538, 0.57833482]])
    kmeans = KMeans(n_clusters=3, init=centroids, random_state=0)
    Y = kmeans.fit_predict(X)

    # Get the final centroids
    centroids1 = kmeans.cluster_centers_

    # cluster_names = {0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3'}
    output_names = {0: 'Good', 1: 'Intermediate', 2: 'Worst'}

    # Add cluster and output labels to the dataframe
    # clean_data['Cluster'] = [cluster_names[cluster] for cluster in Y]
    clean_data['Output'] = [output_names[cluster] for cluster in Y]

    # Save the dataframe to a CSV file
    clean_data.to_csv('Final.csv', index=False)
except:
    pass

def match():
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
    df=pd.read_csv("Final.csv")
    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        state=None
        count_deadlift=0
        try:
            while not stop_button_pressed:
                # print("Hello World")
                _,frame=cap.read()
                
                if not _:
                    st.write("The viseo is ended")
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
                coorx=(int((landmarks[12].x)*width))
                coory=(int((landmarks[12].y)*height))
                coorx1=(int((landmarks[14].x)*width))
                coory1=(int((landmarks[14].y)*height))
                # print(is_point_found)
                if is_point_found:
                    # print(coorx)
                    # print(coory)
                    # print(coorx1)
                    # print(coory1)
                    filtered_df = df[(df['x12'] == landmarks[11].x) & (df['y12'] == landmarks[11].y)]
                    output=filtered_df['Output'].iloc[0]

                    # print(int(landmarks[12].y*height))
                    
                    if(output=="Intermediate"):
                        cv2.line(image,(coorx,coory),(coorx1,coory1),color=(255,0,0), thickness=5)
                        cv2.circle(image,(coorx,coory),color=(255,0,0),radius=20, thickness=-1)
                    elif(output=="Worst"):
                        cv2.line(image,(coorx,coory),(coorx1,coory1),color=(0,0,255), thickness=5)
                        cv2.circle(image,(coorx,coory),color=(0,0,255),radius=20, thickness=-1)
                    elif(output=="Good"):
                        cv2.line(image,(coorx,coory),(coorx1,coory1),color=(0,255,0), thickness=5)
                        cv2.circle(image,(coorx,coory),color=(0,255,0),radius=20, thickness=-1)
                    
                    # print(df['Output'])

                    mp_drawing.draw_landmarks(image,result.pose_landmarks,mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(0,0,0),thickness=2,circle_radius=2),mp_drawing.DrawingSpec(color=(255,255,255),thickness=2,circle_radius=2))
                    
                    # print("Line Drawn")
                    cv2.waitKey(0) 
                    
                else:
                    mp_drawing.draw_landmarks(image,result.pose_landmarks,mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(0,0,0),thickness=2,circle_radius=2),mp_drawing.DrawingSpec(color=(255,255,255),thickness=2,circle_radius=2))
                    # mp_drawing.draw_landmarks(image,result.pose_landmarks,mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),mp_drawing.DrawingSpec(color=(120,50,20),thickness=2,circle_radius=2))
                    # print("Sorry")
                    # mp_drawing.draw_landmarks(image,result.pose_landmarks,mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),mp_drawing.DrawingSpec(color=(45,123,52),thickness=2,circle_radius=2))
                    pass
                # print(df["x12"])
                
                frame_placeholder.image(image,channels="BGR")

                # cv2.imshow("Video Feed",image)
                if(cv2.waitKey(10) & 0xFF== ord('q')) or stop_button_pressed:
                    break
        except:
            # print(df['x12'])
            pass
cap.release()
match()
