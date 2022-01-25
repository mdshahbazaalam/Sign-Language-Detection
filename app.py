from flask import Flask, render_template, Response

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import mediapipe as mp

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Initializing mediapipe holistic

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Detect keypoints from frame

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return results, image

# Draw landmarks from the detected key points

def draw_landmarks(image,results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(200,0,50), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(160,0,0), thickness=1, circle_radius=1))
    
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                             mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))
    
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                             mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                             mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))

# Funtction to store keypoints as NumPy array

def extract_keypoints(results):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh,rh])

# Initializing important variables

actions = np.array(['HELLO','HEY','THANK YOU','GOOD','MORNING'])

DATA_PATH = os.path.join('DATA')
no_of_videos = 30
no_of_frames = 30

# Label-map creation

label_map = {label:num for num, label in enumerate(actions)}

# Set-up neural Network

def setup_neuralnet(actions):
    model = Sequential()
    model.add(LSTM(64,return_sequences = True, activation = 'relu', input_shape = (30,1662)))
    model.add(LSTM(128, return_sequences = True, activation = 'relu'))
    model.add(LSTM(64, return_sequences = False, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(actions.shape[0],activation = 'softmax'))
    return model

# Create model

model = setup_neuralnet(actions)
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

# Load model

model.load_weights('model.h5')

# Real-Time prediction

def real_time_detection(model,actions):
    video = []
    sentence = []
    threshold = 0.4
    camera = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence = 0.7, min_tracking_confidence = 0.7) as holistic:
        while camera.isOpened():
            success, frame = camera.read()
    
            results, image = mediapipe_detection(frame,holistic)
        
            draw_landmarks(image, results)
            
            keypoints = extract_keypoints(results)
            video.append(keypoints)
            video = video[-30:]
            if results.left_hand_landmarks or results.right_hand_landmarks:
                if len(video) == 30:
                    res = model.predict(np.expand_dims(video,axis=0))[0]
            
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if sentence[-1] != actions[np.argmax(res)]:
                                sentence.append(actions[np.argmax(res)])  
                        else:
                            sentence.append(actions[np.argmax(res)])
                    
                        if len(sentence) > 5:
                            sentence = sentence[-5:]
                
                
            cv2.rectangle(image,(0,0),(640,40),(245,116,17),-1)
                
            cv2.putText(image, ' '.join(sentence),(3,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),
                           2,cv2.LINE_AA)
    
            r,buff = cv2.imencode('.jpg',image)
            img = buff.tobytes()
            yield(b'--frame\r\n' b'content-type: image/jpeg\r\n\r\n' + img + b'\r\n')


@app.route('/video')
def video():
    return Response(real_time_detection(model,actions),mimetype='multipart/x-mixed-replace; boundary=frame')

    
if __name__ == '__main__':
    app.run(debug=True)