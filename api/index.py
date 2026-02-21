import os
import sys

# 1. Silence all ML logs and framework noise immediately
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONUNBUFFERED'] = '1'

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Stress Detection API")

# 2. Dynamic path discovery (Serverless safe)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..', 'Stress-Detection-using-ML-and-Image-Processing-Techniques-main')

@app.get("/")
async def health_check():
    """Health check route to resolve 404 and verify file system."""
    return {
        "status": "online",
        "service": "Stress Detection AI",
        "version": "1.0.0",
        "diagnostics": {
            "project_dir_found": os.path.exists(PROJECT_ROOT),
            "model_found": os.path.exists(os.path.join(PROJECT_ROOT, 'model.h5'))
        }
    }

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(status_code=204, content=None)

class PhysiologicalData(BaseModel):
    ecg: float
    emg: float
    foot_gsr: float
    hand_gsr: float
    hr: float
    resp: float

@app.post("/predict_stress")
async def predict_stress(data: PhysiologicalData):
    """Predict physiological stress using KNN."""
    try:
        # Lazy imports to keep execution time low
        import numpy as np
        import pandas as pd
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn import preprocessing
        
        DATA_PATH = os.path.join(PROJECT_ROOT, 'media', 'stress_data.xlsx')
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data file missing at {DATA_PATH}")
            
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
        
        return {"prediction": int(prediction[0]), "label": "Stressed" if prediction[0] == 1 else "Not Stressed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...)):
    """Predict emotion from facial image using CNN."""
    try:
        import numpy as np
        import cv2
        import tensorflow as tf
        
        # Load architecture inside function
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        
        MODEL_PATH = os.path.join(PROJECT_ROOT, 'model.h5')
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model weights missing")
        model.load_weights(MODEL_PATH)
        
        # Process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        XML_PATH = os.path.join(PROJECT_ROOT, 'haarcascade_frontalface_default.xml')
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
                "confidence": float(np.max(prediction))
            })
            
        return {"faces_detected": len(results), "emotions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
