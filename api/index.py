import os
import sys

# Standard Vercel entry point for Django
# 1. Add project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), '..', 'Stress-Detection-using-ML-and-Image-Processing-Techniques-main')
sys.path.append(project_root)

# 2. Add the Django project settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "StressDetection.settings")

# 3. Import WSGI handler
from django.core.wsgi import get_wsgi_application

# Vercel looks for 'app' or 'application'
app = get_wsgi_application()
