from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .models import Cattle, TestResult
from django.contrib.auth.models import User
from .utils import run_prediction
from django.core.paginator import Paginator
from django.db.models import Count, Q
from datetime import datetime
from django.core.files.storage import default_storage
from django.db.models.functions import TruncDate
from django.http import JsonResponse
import torch
from torchvision import transforms
from PIL import Image
import os
import logging

# Import the fallback explainability function
from .explainability_fallback import generate_explainability_images_fallback

# Set up logging
logger = logging.getLogger(__name__)

# Load the ONNX model with better error handling
try:
    import onnxruntime as ort
    model_path = os.path.join(os.path.dirname(__file__), '..', 'cattle_disease_model.onnx')
    if os.path.exists(model_path):
        model = ort.InferenceSession(model_path)
        logger.info("ONNX model loaded successfully")
    else:
        logger.warning(f"ONNX model not found at {model_path}")
        model = None
except Exception as e:
    logger.error(f"Error loading ONNX model: {e}")
    model = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Account created successfully! Please log in.')
            return redirect('login')
        else:
            messages.error(request, 'There was an error with your submission.')
    else:
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})

@login_required
def dashboard(request):
    return render(request, 'detection/dashboard.html')

def dashboard_data(request):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required'}, status=401)

    user = request.user
    
    # Get user's cattle and test results
    user_cattle = Cattle.objects.filter(owner=user)
    user_test_results = TestResult.objects.filter(cattle__owner=user)
    
    cattle_count = user_cattle.count()
    total_tests = user_test_results.count()
    healthy_count = user_test_results.filter(is_healthy=True).count()
    infected_count = user_test_results.filter(is_infected=True).count()
    foot_infections = user_test_results.filter(foot_infection=True).count()
    mouth_infections = user_test_results.filter(mouth_infection=True).count()

    # Get trend data
    trend_data = (
        user_test_results
        .values('tested_at__date')
        .annotate(
            healthy_count=Count('id', filter=Q(is_healthy=True)),
            foot_count=Count('id', filter=Q(foot_infection=True)),
            mouth_count=Count('id', filter=Q(mouth_infection=True))
        )
        .order_by('tested_at__date')
    )

    trend_dates = [entry['tested_at__date'].strftime('%Y-%m-%d') for entry in trend_data]
    trend_healthy = [entry['healthy_count'] for entry in trend_data]
    trend_foot = [entry['foot_count'] for entry in trend_data]
    trend_mouth = [entry['mouth_count'] for entry in trend_data]

    # Provide default data if no trends exist
    if not trend_dates:
        from datetime import date
        trend_dates = [date.today().strftime('%Y-%m-%d')]
        trend_healthy = [healthy_count]
        trend_foot = [foot_infections]
        trend_mouth = [mouth_infections]

    data = {
        'cattle_count': cattle_count,
        'total_tests': total_tests,
        'healthy_count': healthy_count,
        'infected_count': infected_count,
        'foot_infections': foot_infections,
        'mouth_infections': mouth_infections,
        'trend_dates': trend_dates,
        'trend_healthy': trend_healthy,
        'trend_foot': trend_foot,
        'trend_mouth': trend_mouth,
    }
    return JsonResponse(data)

@login_required
def upload_image(request):
    if request.method == 'POST':
        image = request.FILES.get('image')
        cattle_id = request.POST.get('cattle_id')
        
        if not image or not cattle_id:
            messages.error(request, "Please select both an image and cattle.")
            return redirect('upload_image')
            
        try:
            # Verify cattle ownership
            cattle = Cattle.objects.get(id=cattle_id, owner=request.user)
            
            # Create TestResult instance
            test_result = TestResult.objects.create(
                cattle=cattle,
                image=image,
                user=request.user
            )
            
            # Process the image
            success = process_image_and_predict(test_result)
            
            if success:
                messages.success(request, "Image processed successfully!")
                return redirect('results', test_id=test_result.id)
            else:
                messages.error(request, "Error processing image. Please try again.")
                test_result.delete()
                
        except Cattle.DoesNotExist:
            messages.error(request, "Invalid cattle selection.")
        except Exception as e:
            logger.error(f"Error in upload_image: {e}")
            messages.error(request, f"Error: {str(e)}")
    
    cattle = Cattle.objects.filter(owner=request.user)
    return render(request, 'detection/upload.html', {'cattle': cattle})

def process_image_and_predict(test_result):
    """
    Process the uploaded image and run prediction
    """
    try:
        # Get the image path
        image_path = test_result.image.path
        
        # Run basic prediction using utils
        prediction = run_prediction(image_path)
        
        # Update TestResult with prediction results
        test_result.foot_infection = prediction.get('infected_foot', False)
        test_result.mouth_infection = prediction.get('infected_mouth', False)
        test_result.is_infected = prediction.get('is_infected', False)
        test_result.is_healthy = prediction.get('is_healthy', True)
        
        # Generate explainability images using fallback method
        try:
            generate_explainability_images_fallback(image_path, test_result)
            logger.info(f"Generated explainability images for test result {test_result.id}")
        except Exception as e:
            logger.error(f"Error generating explainability images: {e}")
        
        test_result.save()
        return True
        
    except Exception as e:
        logger.error(f"Error in process_image_and_predict: {e}")
        return False

@login_required
def results(request, test_id):
    try:
        test_result = TestResult.objects.get(id=test_id, user=request.user)
        context = {
            'test_result': test_result,
            'image_url': test_result.image.url,
            'integrated_gradients_url': test_result.integrated_gradients_image.url if test_result.integrated_gradients_image else None,
            'gradcam_url': test_result.gradcam_image.url if test_result.gradcam_image else None,
            'occlusion_url': test_result.occlusion_image.url if test_result.occlusion_image else None,
            'lime_url': test_result.lime_image.url if test_result.lime_image else None,
        }
        return render(request, 'detection/results.html', context)
    except TestResult.DoesNotExist:
        messages.error(request, "Test result not found.")
        return redirect('dashboard')

@login_required
def history(request):
    test_results = TestResult.objects.filter(user=request.user).order_by('-tested_at')
    paginator = Paginator(test_results, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request, 'detection/history.html', {'page_obj': page_obj})

@login_required
def register_cattle(request):
    if request.method == 'POST':
        tag_number = request.POST.get('tag_number')
        breed = request.POST.get('breed')
        age = request.POST.get('age')
        
        if not all([tag_number, breed, age]):
            messages.error(request, "All fields are required.")
            return render(request, 'detection/register_cattle.html')
            
        try:
            # Check if tag number already exists for this user
            if Cattle.objects.filter(tag_number=tag_number, owner=request.user).exists():
                messages.error(request, "A cattle with this tag number already exists.")
                return render(request, 'detection/register_cattle.html')
                
            Cattle.objects.create(
                tag_number=tag_number,
                breed=breed,
                age=int(age),
                owner=request.user
            )
            messages.success(request, "Cattle registered successfully.")
            return redirect('dashboard')
        except ValueError:
            messages.error(request, "Age must be a valid number.")
        except Exception as e:
            logger.error(f"Error registering cattle: {e}")
            messages.error(request, f"Error: {str(e)}")
            
    return render(request, 'detection/register_cattle.html')

def register_user(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful.")
            return redirect('dashboard')
        else:
            messages.error(request, "Registration failed. Please check the form.")
    else:
        form = UserCreationForm()
    return render(request, 'detection/register.html', {'form': form})

@login_required
def admin_dashboard(request):
    if not request.user.is_superuser:
        messages.error(request, "You do not have permission to access this page.")
        return redirect('dashboard')
        
    users = User.objects.all().order_by('-date_joined')
    test_results = TestResult.objects.all().order_by('-tested_at')[:50]  # Limit for performance
    
    # Get some statistics
    total_users = users.count()
    total_tests = TestResult.objects.count()
    total_cattle = Cattle.objects.count()
    
    context = {
        'users': users,
        'test_results': test_results,
        'total_users': total_users,
        'total_tests': total_tests,
        'total_cattle': total_cattle,
    }
    return render(request, 'detection/admin_dashboard.html', context)
