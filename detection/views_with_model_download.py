import os
from django.shortcuts import render, redirect
from django.contrib import messages
from .model_downloader import download_model

def upload_view(request):
    # Ensure model is available
    if not os.path.exists('cattle_disease_model.onnx'):
        if not download_model():
            messages.error(request, 'AI model not available. Please try again later.')
            return redirect('dashboard')
    
    # Your existing upload logic here
    if request.method == 'POST':
        # Handle file upload and prediction
        pass
    
    return render(request, 'detection/upload.html')
