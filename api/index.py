import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # 1. Dynamically find the project directory
    # The structure is: /api/index.py
    # Project is in: /Stress-Detection-using-ML-and-Image-Processing-Techniques-main/
    base_path = os.path.dirname(os.path.dirname(__file__))
    project_dir = 'Stress-Detection-using-ML-and-Image-Processing-Techniques-main'
    project_path = os.path.join(base_path, project_dir)
    
    if project_path not in sys.path:
        sys.path.insert(0, project_path)
    
    # 2. Add the inner package path to sys.path too
    inner_path = os.path.join(project_path, 'StressDetection')
    if project_path not in sys.path:
        sys.path.insert(0, project_path)

    logger.info(f"System paths updated. Project path: {project_path}")

    # 3. Set standard Django environment variables
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "StressDetection.settings")

    # 4. Initialize WSGI
    from django.core.wsgi import get_wsgi_application
    application = get_wsgi_application()
    
    # Vercel looks for 'app' or 'application'
    app = application
    logger.info("WSGI application successfully loaded.")

except Exception as e:
    logger.error(f"CRITICAL: Django initialization failed: {e}")
    import traceback
    error_detail = traceback.format_exc()
    logger.error(error_detail)

    # Fallback to show error in browser for easier debugging
    def app(environ, start_response):
        status = '500 Internal Server Error'
        body = f"""
        <html>
            <body style='font-family: sans-serif; padding: 40px;'>
                <h1 style='color: #d9534f;'>Django Initialization Failed</h1>
                <p>The serverless function failed to start the Django application.</p>
                <div style='background: #f8f9fa; padding: 20px; border: 1px solid #ddd; border-radius: 5px;'>
                    <strong>Error:</strong> {e}
                    <br><br>
                    <strong>Traceback:</strong>
                    <pre style='font-size: 12px; color: #555;'>{error_detail}</pre>
                </div>
            </body>
        </html>
        """.encode('utf-8')
        response_headers = [('Content-type', 'text/html'), ('Content-Length', str(len(body)))]
        start_response(status, response_headers)
        return [body]

# Vercel compatibility alias
application = app
