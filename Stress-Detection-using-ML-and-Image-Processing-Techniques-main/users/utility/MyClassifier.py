import os
import httpx
import pandas as pd
from django.conf import settings

class KNNclassifier:
    def __init__(self):
        self.backend_url = os.environ.get("ML_BACKEND_URL")

    def getKnnResults(self):
        """Fetch pre-calculated KNN metrics and samples from the Render ML Engine."""
        if not self.backend_url:
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0

        try:
            response = httpx.get(f"{self.backend_url}/knn_results", timeout=60.0)
            if response.status_code == 200:
                data = response.json()
                
                # Reconstruct DataFrame from samples
                df = pd.DataFrame(data["samples"])
                
                return (
                    df,
                    data["accuracy"],
                    data["classificationerror"],
                    data["sensitivity"],
                    data["Specificity"],
                    data["fsp"],
                    data["precision"]
                )
            else:
                print(f"Backend Error: {response.status_code}")
                return pd.DataFrame(), 0, 0, 0, 0, 0, 0
        except Exception as e:
            print(f"Network Error: {str(e)}")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0