import os
import sys
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Stress Detection API (Proxy Mode)")

# Configuration
ML_BACKEND_URL = os.environ.get("ML_BACKEND_URL") # e.g., https://your-ml-api.onrender.com
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Stress-Detection-using-ML-and-Image-Processing-Techniques-main')

@app.get("/")
async def health_check():
    return {
        "status": "online",
        "mode": "Proxy" if ML_BACKEND_URL else "Local (Standalone)",
        "backend_url": ML_BACKEND_URL,
        "diagnostics": {
            "project_dir_found": os.path.exists(PROJECT_ROOT)
        }
    }

async def proxy_request(method: str, path: str, content=None, headers=None, params=None):
    if not ML_BACKEND_URL:
        raise HTTPException(status_code=503, detail="ML Backend not configured and local execution exceeded limits.")
    
    url = f"{ML_BACKEND_URL.rstrip('/')}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.request(method, url, content=content, headers=headers, params=params)
            return JSONResponse(status_code=response.status_code, content=response.json())
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Error connecting to ML Backend: {str(e)}")

@app.post("/predict_stress")
async def predict_stress(request: Request):
    if ML_BACKEND_URL:
        body = await request.body()
        return await proxy_request("POST", "predict_stress", content=body, headers={"Content-Type": "application/json"})
    
    # Fallback to local (only if dependencies are magically fixed or slimmed)
    # [Rest of local logic from previous implementation...]
    return {"error": "Local ML not available. Please deploy ML Backend to Render/Railway."}

@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...)):
    if ML_BACKEND_URL:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{ML_BACKEND_URL}/predict_emotion", files=files)
            return JSONResponse(status_code=response.status_code, content=response.json())
    
    return {"error": "Local ML not available. Please deploy ML Backend to Render/Railway."}
