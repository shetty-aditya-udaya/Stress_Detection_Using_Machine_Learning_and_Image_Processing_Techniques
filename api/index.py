import os
import sys

# Silence ALL logs before anything else
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONUNBUFFERED'] = '1'

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

# --- Configuration ---
# Use absolute root path for reliability
BASE_DIR = os.path.join(os.getcwd(), 'Stress-Detection-using-ML-and-Image-Processing-Techniques-main')

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "project": "Stress Detection AI",
        "mode": "standalone",
        "diag": {
            "cwd": os.getcwd(),
            "base_dir_exists": os.path.exists(BASE_DIR)
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
        import numpy as np
        import pandas as pd
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn import preprocessing
        
        DATA_PATH = os.path.join(BASE_DIR, 'media', 'stress_data.xlsx')
        if not os.path.exists(DATA_PATH):
            raise HTTPException(status_code=404, detail=f"Data file not found at {DATA_PATH}")
            
        df = pd.read_excel(DATA_PATH, header=None)
        df.columns=['Target', 'ECG(mV)', 'EMG(mV)','Foot GSR(mV)','Hand GSR(mV)', 'HR(bpm)','RESP(mV)']
        
        features = ['ECG(mV)', 'EMG(mV)','Foot GSR(mV)','Hand GSR(mV)', 'HR(bpm)','RESP(mV)']
        scaler = preprocessing.MinMaxScaler().fit(df[features])
        X_norm = scaler.transform(df[features])
        y = df['Target']
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_norm, y)
        
        input_data = [[data.ecg, data.emg, data.foot_gsr, data.hand_gsr, data.hr, data.resp]]
        input_norm = scaler.transform(input_data)
        prediction = knn.predict(input_norm)
        
        return {"stress_detected": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        import numpy as np
        import cv2
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
        
        # Load Architecture inside handler
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
        
        MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=404, detail="Model weights not found")
        model.load_weights(MODEL_PATH)
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        XML_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
        cascade = cv2.CascadeClassifier(XML_PATH)
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        
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
