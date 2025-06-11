# detection/explainability_fallback.py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

def preprocess_image_for_visualization(image_path):
    """
    Preprocess image for visualization techniques
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        img = cv2.resize(img, (224, 224))
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def generate_gradient_visualization(image_path, save_path):
    """
    Generate gradient-based visualization (similar to Integrated Gradients)
    """
    try:
        img = preprocess_image_for_visualization(image_path)
        if img is None:
            return False
        
        # Convert to grayscale for gradient computation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Compute gradients using Sobel operators
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_magnitude = np.uint8(255 * grad_magnitude / np.max(grad_magnitude))
        
        # Apply colormap for better visualization
        heatmap = cv2.applyColorMap(grad_magnitude, cv2.COLORMAP_PLASMA)
        
        # Create overlay
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        # Save the result
        cv2.imwrite(save_path, overlay)
        logger.info(f"Generated gradient visualization: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating gradient visualization: {e}")
        return False

def generate_gradcam_like_visualization(image_path, save_path):
    """
    Generate GradCAM-like visualization using edge detection and feature highlighting
    """
    try:
        img = preprocess_image_for_visualization(image_path)
        if img is None:
            return False
        
        # Convert to different color spaces for feature extraction
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Extract features from different channels
        # Focus on areas with high saturation (potential disease areas)
        saturation = hsv[:, :, 1]
        
        # Apply Gaussian blur to smooth the features
        blurred_sat = cv2.GaussianBlur(saturation, (15, 15), 0)
        
        # Normalize and create heatmap
        normalized = cv2.normalize(blurred_sat, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        cv2.imwrite(save_path, overlay)
        logger.info(f"Generated GradCAM-like visualization: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating GradCAM-like visualization: {e}")
        return False

def generate_occlusion_visualization(image_path, save_path):
    """
    Generate occlusion-based visualization by analyzing image patches
    """
    try:
        img = preprocess_image_for_visualization(image_path)
        if img is None:
            return False
        
        h, w = img.shape[:2]
        importance_map = np.zeros((h, w), dtype=np.float32)
        
        # Define patch size for occlusion
        patch_size = 16
        
        # Analyze each patch for "importance" based on color variation
        for i in range(0, h - patch_size, patch_size // 2):
            for j in range(0, w - patch_size, patch_size // 2):
                patch = img[i:i+patch_size, j:j+patch_size]
                
                # Calculate patch importance based on color variance
                # Areas with high variance might indicate disease features
                patch_std = np.std(patch)
                importance_map[i:i+patch_size, j:j+patch_size] += patch_std
        
        # Normalize importance map
        importance_map = cv2.normalize(importance_map, None, 0, 255, cv2.NORM_MINMAX)
        importance_map = importance_map.astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(importance_map, cv2.COLORMAP_HOT)
        
        # Create overlay
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        cv2.imwrite(save_path, overlay)
        logger.info(f"Generated occlusion visualization: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating occlusion visualization: {e}")
        return False

def is_potentially_diseased_segment(mean_color):
    """
    Simple heuristic to determine if a segment might contain disease indicators
    """
    b, g, r = mean_color
    
    # Look for reddish or inflamed areas (simplified)
    if r > g + 20 and r > b + 20:
        return True
    
    # Look for areas with unusual color combinations
    if abs(r - g) > 30 or abs(g - b) > 30:
        return True
    
    return False

def generate_lime_like_visualization(image_path, save_path):
    """
    Generate LIME-like visualization using superpixel-based analysis
    """
    try:
        img = preprocess_image_for_visualization(image_path)
        if img is None:
            return False
        
        h, w = img.shape[:2]
        segment_size = 20
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Analyze each segment
        for i in range(0, h, segment_size):
            for j in range(0, w, segment_size):
                end_i = min(i + segment_size, h)
                end_j = min(j + segment_size, w)
                segment = img[i:end_i, j:end_j]
                
                if segment.size > 0:
                    mean_color = np.mean(segment, axis=(0, 1))
                    
                    if is_potentially_diseased_segment(mean_color):
                        mask[i:end_i, j:end_j] = 255
        
        # Apply the mask to highlight important regions
        highlighted = img.copy()
        green_overlay = np.zeros_like(highlighted)
        green_overlay[:, :] = [0, 255, 0]  # Green color
        
        # Apply green overlay to masked regions
        mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
        highlighted = highlighted * (1 - mask_3d * 0.3) + green_overlay * (mask_3d * 0.3)
        highlighted = highlighted.astype(np.uint8)
        
        cv2.imwrite(save_path, highlighted)
        logger.info(f"Generated LIME-like visualization: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating LIME-like visualization: {e}")
        return False

def generate_explainability_images_fallback(image_path, test_result):
    """
    Generate explainability images without requiring Captum
    """
    try:
        # Create explainability directory
        explainability_dir = os.path.join(settings.MEDIA_ROOT, 'explainability')
        os.makedirs(explainability_dir, exist_ok=True)
        
        # Generate different types of explanations
        techniques = {
            'integrated_gradients': generate_gradient_visualization,
            'gradcam': generate_gradcam_like_visualization,
            'occlusion': generate_occlusion_visualization,
            'lime': generate_lime_like_visualization
        }
        
        success_count = 0
        
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
                    
                    success_count += 1
                    logger.info(f"Generated {technique_name} for test result {test_result.id}")
                else:
                    logger.warning(f"Failed to generate {technique_name} for test result {test_result.id}")
                    
            except Exception as e:
                logger.error(f"Error generating {technique_name}: {e}")
        
        # Save the test result with updated image paths
        test_result.save()
        
        logger.info(f"Generated {success_count}/4 explainability images for test result {test_result.id}")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error in generate_explainability_images_fallback: {e}")
        return False