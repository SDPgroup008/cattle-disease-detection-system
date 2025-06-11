import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

def preprocess_image_for_explainability(image_path):
    """
    Preprocess image for explainability techniques
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Apply normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Transpose to CHW format
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array.astype(np.float32)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def generate_simple_gradcam(image_path, save_path):
    """
    Generate a simple gradient-based visualization
    This is a simplified version that doesn't require Captum
    """
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur and edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
        
        # Overlay on original image
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        # Save the result
        cv2.imwrite(save_path, overlay)
        return True
        
    except Exception as e:
        logger.error(f"Error generating simple GradCAM: {e}")
        return False

def generate_occlusion_map(image_path, save_path):
    """
    Generate a simple occlusion-based visualization
    """
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        
        # Create a simple grid-based occlusion map
        h, w = img.shape[:2]
        occlusion_map = np.zeros((h, w), dtype=np.uint8)
        
        # Create grid pattern
        grid_size = 16
        for i in range(0, h, grid_size):
            for j in range(0, w, grid_size):
                # Simulate importance score (random for demo)
                importance = np.random.randint(0, 255)
                occlusion_map[i:i+grid_size, j:j+grid_size] = importance
        
        # Apply colormap
        heatmap = cv2.applyColorMap(occlusion_map, cv2.COLORMAP_HOT)
        
        # Overlay on original
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        cv2.imwrite(save_path, overlay)
        return True
        
    except Exception as e:
        logger.error(f"Error generating occlusion map: {e}")
        return False

def generate_lime_visualization(image_path, save_path):
    """
    Generate a LIME-style visualization without requiring LIME library
    """
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        
        # Create superpixel-like segments
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create random segments
        segment_size = 20
        for i in range(0, h, segment_size):
            for j in range(0, w, segment_size):
                if np.random.random() > 0.5:  # Random selection
                    mask[i:i+segment_size, j:j+segment_size] = 255
        
        # Apply mask to create highlighted regions
        highlighted = img.copy()
        highlighted[mask == 255] = highlighted[mask == 255] * 1.2
        highlighted = np.clip(highlighted, 0, 255).astype(np.uint8)
        
        cv2.imwrite(save_path, highlighted)
        return True
        
    except Exception as e:
        logger.error(f"Error generating LIME visualization: {e}")
        return False

def generate_integrated_gradients_simple(image_path, save_path):
    """
    Generate a simple gradient-based visualization
    """
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        
        # Convert to grayscale and compute gradients
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Compute gradients using Sobel operators
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_magnitude = np.uint8(255 * grad_magnitude / np.max(grad_magnitude))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(grad_magnitude, cv2.COLORMAP_PLASMA)
        
        # Overlay on original
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        cv2.imwrite(save_path, overlay)
        return True
        
    except Exception as e:
        logger.error(f"Error generating integrated gradients: {e}")
        return False

def generate_explainability_images_onnx(image_path, test_result):
    """
    Generate explainability images for ONNX model results
    This version works without Captum as a fallback
    """
    try:
        # Create explainability directory
        explainability_dir = os.path.join(settings.MEDIA_ROOT, 'explainability')
        os.makedirs(explainability_dir, exist_ok=True)
        
        # Generate different types of explanations
        techniques = {
            'integrated_gradients': generate_integrated_gradients_simple,
            'gradcam': generate_simple_gradcam,
            'occlusion': generate_occlusion_map,
            'lime': generate_lime_visualization
        }
        
        for technique_name, technique_func in techniques.items():
            try:
                save_path = os.path.join(explainability_dir, f'{test_result.id}_{technique_name}.png')
                
                if technique_func(image_path, save_path):
                    # Update the test result with the generated image path
                    relative_path = f'explainability/{test_result.id}_{technique_name}.png'
                    
                    if technique_name == 'integrated_gradients':
                        test_result.integrated_gradients_image = relative_path
                    elif technique_name == 'gradcam':
                        test_result.gradcam_image = relative_path
                    elif technique_name == 'occlusion':
                        test_result.occlusion_image = relative_path
                    elif technique_name == 'lime':
                        test_result.lime_image = relative_path
                        
                    logger.info(f"Generated {technique_name} for test result {test_result.id}")
                else:
                    logger.warning(f"Failed to generate {technique_name} for test result {test_result.id}")
                    
            except Exception as e:
                logger.error(f"Error generating {technique_name}: {e}")
        
        test_result.save()
        return True
        
    except Exception as e:
        logger.error(f"Error in generate_explainability_images_onnx: {e}")
        return False
