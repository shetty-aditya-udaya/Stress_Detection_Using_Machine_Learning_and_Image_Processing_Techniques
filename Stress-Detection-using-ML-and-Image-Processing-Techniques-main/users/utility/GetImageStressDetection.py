import os
import httpx
from django.conf import settings

class ImageExpressionDetect:
    def __init__(self):
        self.backend_url = os.environ.get("ML_BACKEND_URL")

    def getExpression(self, imagepath):
        """Perform facial emotion detection via remote Render ML Engine."""
        if not self.backend_url:
            return "ML Backend URL Not Configured"

        # Construct full path to the uploaded image in media/
        filepath = os.path.join(settings.MEDIA_ROOT, imagepath)
        
        if not os.path.exists(filepath):
            return f"Error: Image not found at {filepath}"

        try:
            with open(filepath, 'rb') as f:
                files = {'file': (imagepath, f, 'image/jpeg')}
                response = httpx.post(f"{self.backend_url}/predict_emotion", files=files, timeout=60.0)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("emotions"):
                        # Just return the first detected emotion for compatibility with the legacy UI
                        return data["emotions"][0]["emotion"]
                    return "No faces detected"
                return f"Backend Error: {response.status_code}"
        except Exception as e:
            return f"Network Error: {str(e)}"

    def getLiveDetect(self):
        """Legacy method for live camera detection. 
        Note: OpenCV camera capture behaves differently in cloud environments and usually requires WebRTC."""
        print("Live detection triggered. In cloud environments, use a client-side WebRTC solution.")
        return "Not available in cloud deployment"
