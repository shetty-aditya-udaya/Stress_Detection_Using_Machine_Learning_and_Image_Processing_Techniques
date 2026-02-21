import os
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Stress Detection Proxy (Vercel)")

# Configuration - Must be set in Vercel Dashboard
ML_BACKEND_URL = os.environ.get("ML_BACKEND_URL")

@app.get("/")
async def health_check():
    """Proxy health check to satisfy Vercel status."""
    return {
        "status": "online",
        "role": "Proxy Only",
        "backend_url_configured": bool(ML_BACKEND_URL),
        "note": "All ML inference is handled by the Render backend."
    }

async def forward_request(method: str, path: str, content=None, headers=None, params=None):
    if not ML_BACKEND_URL:
        raise HTTPException(
            status_code=503, 
            detail="ML_BACKEND_URL is not configured in Vercel. Please add it to your environment variables."
        )
    
    url = f"{ML_BACKEND_URL.rstrip('/')}/{path.lstrip('/')}"
    # Use a longer timeout for ML inference
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.request(
                method, 
                url, 
                content=content, 
                headers=headers, 
                params=params
            )
            # Return JSON exactly as received from Render
            return JSONResponse(status_code=response.status_code, content=response.json())
        except httpx.ConnectError:
            raise HTTPException(status_code=502, detail="Unable to connect to ML Backend on Render.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_stress")
async def predict_stress(request: Request):
    """Proxy for physiological stress prediction."""
    body = await request.body()
    return await forward_request(
        "POST", 
        "predict_stress", 
        content=body, 
        headers={"Content-Type": "application/json"}
    )

@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...)):
    """Proxy for facial emotion prediction (multipart)."""
    if not ML_BACKEND_URL:
         raise HTTPException(status_code=503, detail="ML_BACKEND_URL missing.")
         
    # Prepare files for forwarding
    files = {"file": (file.filename, await file.read(), file.content_type)}
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(f"{ML_BACKEND_URL}/predict_emotion", files=files)
            return JSONResponse(status_code=response.status_code, content=response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(status_code=204, content=None)
