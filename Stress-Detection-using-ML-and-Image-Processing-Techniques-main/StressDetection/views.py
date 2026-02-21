import os
import httpx
from django.shortcuts import render
from users.forms import UserRegistrationForm

def home(request):
    """Home view for Vercel deployment. Handles stress prediction form or renders landing page."""
    context = {}
    if request.method == 'POST':
        # Get data from form
        try:
            data = {
                "ecg": float(request.POST.get('ecg', 0)),
                "emg": float(request.POST.get('emg', 0)),
                "foot_gsr": float(request.POST.get('foot_gsr', 0)),
                "hand_gsr": float(request.POST.get('hand_gsr', 0)),
                "hr": float(request.POST.get('hr', 0)),
                "resp": float(request.POST.get('resp', 0))
            }
            
            # Forward to Render
            ML_BACKEND_URL = os.environ.get("ML_BACKEND_URL")
            if ML_BACKEND_URL:
                url = f"{ML_BACKEND_URL.rstrip('/')}/predict_stress"
                try:
                    # Use a longer timeout for potential Render cold starts
                    with httpx.Client(timeout=60.0) as client:
                        response = client.post(url, json=data)
                        if response.status_code == 200:
                            context['result'] = response.json()
                        else:
                            context['error'] = f"ML Backend returned error: {response.status_code}"
                except Exception as e:
                    context['error'] = f"Failed to connect to ML Backend: {str(e)}"
            else:
                context['error'] = "ML_BACKEND_URL is not configured in Vercel."
                
        except (ValueError, TypeError):
            context['error'] = "Please enter valid numerical values for all fields."

    return render(request, 'index.html', context)

def logout(request):
    return render(request, 'index.html', {})

def UserLogin(request):
    return render(request, 'UserLogin.html', {})

def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})

def UserRegister(request):
    form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})
