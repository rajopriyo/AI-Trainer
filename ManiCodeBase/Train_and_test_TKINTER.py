import tkinter as tk
from tkinter import Label, Button
import threading
import pandas as pd
import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans

class VideoProcessor:
    def __init__(self):
        self.cap = cv2.VideoCapture('Rajo.mp4')
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.df = None

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def process_training_video(self):
        listl1 = []
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            try:
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = True
                    result = pose.process(image)
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    landmarks = result.pose_landmarks.landmark
                    templist = []
                    for i in range(33):
                        templist.append(landmarks[i].x)
                        templist.append(landmarks[i].y)
                        templist.append(landmarks[i].z)
                        templist.append(landmarks[i].visibility)
                    listl1.append(templist)
                self.df = pd.DataFrame(listl1, columns=[f'x{i}' for i in range(1, 34)] + [f'y{i}' for i in range(1, 34)] + [f'z{i}' for i in range(1, 34)] + [f'v{i}' for i in range(1, 34)])
                self.df.to_csv('data.csv', index=False)
            except Exception as e:
                print(f"Error processing training video: {e}")

        self.clean_data()

    def clean_data(self):
        try:
            dataFrame = pd.read_csv("data.csv")
            df = dataFrame[['x12', 'y12', 'x14', 'y14']]
            df.to_csv("Cleaneddata.csv", index=False)
            self.cluster_data()
        except Exception as e:
            print(f"Error cleaning data: {e}")

    def cluster_data(self):
        try:
            clean_data = pd.read_csv('Cleaneddata.csv')
            X = clean_data.iloc[:, [0, 1, 2, 3]].values
            kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)  # Explicitly setting n_init
            Y = kmeans.fit_predict(X)
            output_names = {0: 'Good', 1: 'Intermediate', 2: 'Worst'}
            clean_data['Output'] = [output_names[cluster] for cluster in Y]
            clean_data.to_csv('Final.csv', index=False)
        except Exception as e:
            print(f"Error clustering data: {e}")


    def run_test_video(self):
        cap = cv2.VideoCapture("Rajo.mp4")
        df = pd.read_csv("Final.csv")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = True
                    result = pose.process(image)
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    landmarks = result.pose_landmarks.landmark
                    point = {'x12': landmarks[11].x, 'y12': landmarks[11].y, 'x14': landmarks[13].x, 'y14': landmarks[13].y}
                    isContained = (df['x12'] == point['x12']) & (df['y12'] == point['y12']) & (df['x14'] == point['x14']) & (df['y14'] == point['y14'])
                    is_point_found = any(isContained)

                    coorx = (int((landmarks[12].x) * width))
                    coory = (int((landmarks[12].y) * height))
                    coorx1 = (int((landmarks[14].x) * width))
                    coory1 = (int((landmarks[14].y) * height))

                    if is_point_found:
                        filtered_df = df[(df['x12'] == landmarks[11].x) & (df['y12'] == landmarks[11].y)]
                        output = filtered_df['Output'].iloc[0]

                        if output == "Good":
                            cv2.line(image, (coorx, coory), (coorx1, coory1), color=(0, 255, 0), thickness=5)
                            cv2.circle(image, (coorx, coory), color=(0, 255, 0), radius=20, thickness=-1)
                        elif output == "Intermediate":
                            cv2.line(image, (coorx, coory), (coorx1, coory1), color=(255, 0, 0), thickness=5)
                            cv2.circle(image, (coorx, coory), color=(255, 0, 0), radius=20, thickness=-1)
                        else:
                            cv2.line(image, (coorx, coory), (coorx1, coory1), color=(0, 0, 255), thickness=5)
                            cv2.circle(image, (coorx, coory), color=(0, 0, 255), radius=20, thickness=-1)

                    self.mp_drawing.draw_landmarks(image, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, self.mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

                    cv2.imshow("Video Feed", image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            except Exception as e:
                print(f"Error running test video: {e}")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processing Application")
        self.label = Label(root, text="Please wait while the training video is processed...")
        self.label.pack(pady=20)
        self.start_button = Button(root, text="Start", command=self.start_video_processing)
        self.start_button.pack(pady=20)
        self.video_processor = VideoProcessor()
        self.processing_thread = None

    def start_video_processing(self):
        self.start_button.config(state="disabled")
        self.label.config(text="Processing training video...")
        self.processing_thread = threading.Thread(target=self.run_processing)
        self.processing_thread.start()

    def run_processing(self):
        self.video_processor.process_training_video()
        self.label.config(text="Training video processed. Click 'Start' to run the test video.")
        self.start_button.config(state="normal", command=self.run_test_video)

    def run_test_video(self):
        self.start_button.config(state="disabled")
        self.label.config(text="Running test video...")
        self.processing_thread = threading.Thread(target=self.video_processor.run_test_video)
        self.processing_thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

