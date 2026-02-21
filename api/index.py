import os
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Stress Detection API (Vercel Proxy)")

# Configuration
ML_BACKEND_URL = os.environ.get("ML_BACKEND_URL")

@app.get("/")
async def health_check():
    """Lightweight health check for Vercel."""
    return {
        "status": "online",
        "service": "Vercel Proxy",
        "backend_url_configured": bool(ML_BACKEND_URL),
        "note": "This is a lightweight proxy forwarding requests to the ML Engine."
    }

async def proxy_request(method: str, path: str, content=None, headers=None, params=None):
    if not ML_BACKEND_URL:
        raise HTTPException(status_code=503, detail="ML Backend URL not configured in Vercel environment variables.")
    
    url = f"{ML_BACKEND_URL.rstrip('/')}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.request(method, url, content=content, headers=headers, params=params)
            return JSONResponse(status_code=response.status_code, content=response.json())
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Error connecting to ML Backend: {str(e)}")

@app.post("/predict_stress")
async def predict_stress(request: Request):
    body = await request.body()
    return await proxy_request("POST", "predict_stress", content=body, headers={"Content-Type": "application/json"})

@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...)):
    if not ML_BACKEND_URL:
        raise HTTPException(status_code=503, detail="ML Backend URL not configured.")
        
    files = {"file": (file.filename, await file.read(), file.content_type)}
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(f"{ML_BACKEND_URL}/predict_emotion", files=files)
            return JSONResponse(status_code=response.status_code, content=response.json())
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Error connecting to ML Backend: {str(e)}")
