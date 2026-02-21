import os
import sys
import base64
import numpy as np
import pandas as pd
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

app = FastAPI()

# --- Paths & Configuration ---
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'Stress-Detection-using-ML-and-Image-Processing-Techniques-main')
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')
XML_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
DATA_PATH = os.path.join(BASE_DIR, 'media', 'stress_data.xlsx')

# --- Model Loading (Lazy) ---
cnn_model = None
knn_model = None
minmax_scale = None
face_cascade = None

def get_cnn_model():
    global cnn_model
    if cnn_model is None:
        # Recreate the architecture as defined in kerasmodel.py
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        
        if os.path.exists(MODEL_PATH):
            model.load_weights(MODEL_PATH)
        cnn_model = model
    return cnn_model

def get_face_cascade():
    global face_cascade
    if face_cascade is None:
        if os.path.exists(XML_PATH):
            face_cascade = cv2.CascadeClassifier(XML_PATH)
    return face_cascade

def get_knn_resources():
    global knn_model, minmax_scale
    if knn_model is None:
        if os.path.exists(DATA_PATH):
            df = pd.read_excel(DATA_PATH, header=None)
            df.columns=['Target', 'ECG(mV)', 'EMG(mV)','Foot GSR(mV)','Hand GSR(mV)', 'HR(bpm)','RESP(mV)']
            
            features = ['ECG(mV)', 'EMG(mV)','Foot GSR(mV)','Hand GSR(mV)', 'HR(bpm)','RESP(mV)']
            minmax_scale = preprocessing.MinMaxScaler().fit(df[features])
            X_norm = minmax_scale.transform(df[features])
            y = df['Target']
            
            knn_model = KNeighborsClassifier(n_neighbors=5)
            knn_model.fit(X_norm, y)
        else:
            raise Exception(f"Data file not found at {DATA_PATH}")
    return knn_model, minmax_scale

# --- Routes ---

@app.get("/")
async def health_check():
    files = []
    if os.path.exists(BASE_DIR):
        files = os.listdir(BASE_DIR)
    
    return {
        "status": "healthy",
        "project": "Stress Detection AI",
        "base_dir": BASE_DIR,
        "base_dir_exists": os.path.exists(BASE_DIR),
        "files_in_base_dir": files,
        "endpoints": {
            "/": "Health check",
            "/predict_emotion": "POST - Detect emotion from image",
            "/predict_stress": "POST - Detect stress from physiological data"
        }
    }

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})

class PhysiologicalData(BaseModel):
    ecg: float
    emg: float
    foot_gsr: float
    hand_gsr: float
    hr: float
    resp: float

@app.post("/predict_stress")
async def predict_stress(data: PhysiologicalData):
    try:
        knn, scaler = get_knn_resources()
        input_data = [[data.ecg, data.emg, data.foot_gsr, data.hand_gsr, data.hr, data.resp]]
        input_norm = scaler.transform(input_data)
        prediction = knn.predict(input_norm)
        return {"stress_detected": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        cascade = get_face_cascade()
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        model = get_cnn_model()
        
        results = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            results.append({
                "emotion": emotion_dict[maxindex],
                "confidence": float(np.max(prediction)),
                "box": [int(x), int(y), int(w), int(h)]
            })
            
        return {"faces_detected": len(results), "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
